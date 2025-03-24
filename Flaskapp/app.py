# app.py
import os
import json
import re
import pickle
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
class Config:
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    BASE_API_URL = os.getenv('BASE_API_URL', 'http://localhost:5000')  # Will be updated by Render
    API_KEY = os.getenv('API_KEY')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    GENERATION_CONFIG = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Setup logging
logging.basicConfig(
    level=getattr(logging, app.config['LOG_LEVEL']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler = RotatingFileHandler('/var/log/app.log', maxBytes=10000, backupCount=3)
handler.setLevel(getattr(logging, app.config['LOG_LEVEL']))
app.logger.addHandler(handler)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure Google Gemini API
genai.configure(api_key=app.config['API_KEY'])

SYSTEM_INSTRUCTION = (
    "Extract blood report data and provide observed values in JSON format. "
    "Include gender (1 for male, 0 for female) and age from the report. Normalize 'Hemoglobin' and 'Haemoglobin' to 'Hemoglobin'. "
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def upload_to_gemini(path, mime_type="image/jpeg"):
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        return file
    except Exception as e:
        app.logger.error(f"Failed to upload file '{path}': {str(e)}")
        raise RuntimeError(f"Failed to upload file: {str(e)}")

def extract_blood_report(image_path):
    try:
        local_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=app.config['GENERATION_CONFIG'],
            system_instruction=SYSTEM_INSTRUCTION,
        )
        uploaded_file = upload_to_gemini(image_path)
        chat_session = local_model.start_chat(history=[
            {"role": "user", "parts": [uploaded_file]},
            {"role": "model", "parts": [
                """```json
                {
                  "Hemoglobin": "9.0",
                  "Hematocrit (PCV)": "27",
                  "RBC Count": "6.1",
                  "MCV": "44",
                  "MCH": "15",
                  "MCHC": "33",
                  "RDW CV": "20",
                  "RDW SD": "37.4",
                  "Total Leucocyte Count": "8300",
                  "NEUTROPHILS": "78",
                  "LYMPHOCYTES": "16",
                  "EOSINOPHILS": "04",
                  "BASOPHILS": "00",
                  "MONOCYTES": "02",
                  "Platelet Count": "225000",
                  "Mean Platelet Volume (MPV)": "8",
                  "gender": 1,
                  "age": "26"
                }
                ```"""
            ]}
        ])
        response = chat_session.send_message("Extract again")
        result = response.text
        match = re.search(r"```json\n(.*?)\n```", result, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON found in response!")
        json_text = match.group(1)
        data = json.loads(json_text)
        if "Haemoglobin" in data:
            data["Hemoglobin"] = data.pop("Haemoglobin")
        if "age" not in data and "Age" not in data:
            data["age"] = None
        elif "Age" in data:
            data["age"] = data.pop("Age")
        app.logger.info("Successfully extracted blood report data")
        return data
    except Exception as e:
        app.logger.error(f"Error extracting blood report: {str(e)}")
        raise RuntimeError(f"Error extracting blood report: {str(e)}")

def load_anemia_model(model_path='./LRmodel.pkl'):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        app.logger.info("Anemia model loaded successfully")
        return model
    except Exception as e:
        app.logger.error(f"Failed to load anemia model: {str(e)}")
        raise RuntimeError(f"Failed to load anemia model: {str(e)}")

def load_dengue_model_and_scaler(model_path='./dengue.keras', scaler_path='./scaler.pkl'):
    try:
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        app.logger.info("Dengue model and scaler loaded successfully")
        return model, scaler
    except Exception as e:
        app.logger.error(f"Failed to load dengue model/scaler: {str(e)}")
        raise RuntimeError(f"Failed to load dengue model/scaler: {str(e)}")

# Load models
try:
    anemia_model = load_anemia_model()
    dengue_model, dengue_scaler = load_dengue_model_and_scaler()
except Exception as e:
    app.logger.critical(f"Failed to initialize models: {str(e)}")
    raise

# Global variable for latest report
latest_report_data = None

def extract_features_anemia(api_data):
    try:
        hemoglobin = api_data.get("Hemoglobin", api_data.get("Haemoglobin", 5))
        features = {
            'Gender': float(api_data['gender']),
            'Hemoglobin': float(hemoglobin),
            'MCH': float(api_data['MCH']),
            'MCHC': float(api_data['MCHC']),
            'MCV': float(api_data['MCV'])
        }
        return pd.DataFrame([features])
    except (KeyError, ValueError) as e:
        app.logger.error(f"Error extracting anemia features: {str(e)}")
        raise ValueError(f"Error in anemia features: {str(e)}")

def extract_features_dengue(api_data):
    try:
        hemoglobin = api_data.get("Hemoglobin", api_data.get("Haemoglobin"))
        if hemoglobin is None:
            raise ValueError("Hemoglobin value is missing")
        features = [
            float(api_data.get("gender", 1)),
            float(api_data.get("age", 0)),
            float(hemoglobin),
            float(api_data.get("NEUTROPHILS", api_data.get("Neutrophils", 0))),
            float(api_data.get("LYMPHOCYTES", api_data.get("Lymphocytes", 0))),
            float(api_data.get("MONOCYTES", api_data.get("Monocytes", 0))),
            float(api_data.get("EOSINOPHILS", api_data.get("Eosinophils", 0))),
            float(api_data.get("RBC Count", 0)),
            float(api_data.get("Hematocrit (PCV)", api_data.get("Hct", 0))),
            float(api_data.get("MCV", 0)),
            float(api_data.get("MCH", 0)),
            float(api_data.get("MCHC", 0)),
            float(api_data.get("RDW CV", api_data.get("RDW-CV", 0))),
            float(api_data.get("Platelet Count", 0)),
            float(api_data.get("Mean Platelet Volume (MPV)", api_data.get("MPV", 0))),
            float(api_data.get("PDW", 0)),
            float(api_data.get("PCT", 0)),
            float(api_data.get("Total Leucocyte Count", 0))
        ]
        input_data = np.array(features).reshape(1, -1)
        return dengue_scaler.transform(input_data)
    except (KeyError, ValueError) as e:
        app.logger.error(f"Error extracting dengue features: {str(e)}")
        raise ValueError(f"Error in dengue features: {str(e)}")

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': str(error)}), 400

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

# Routes
@app.route('/api/upload', methods=['POST'])
def upload_report():
    global latest_report_data
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: jpg, jpeg, png'}), 400

        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        latest_report_data = extract_blood_report(image_path)
        os.remove(image_path)
        
        return jsonify({'message': 'Image uploaded and processed successfully', 'data': latest_report_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/report', methods=['GET'])
def get_report():
    global latest_report_data
    try:
        if latest_report_data is None:
            return jsonify({'error': 'No report data available. Please upload an image first.'}), 400
        return jsonify(latest_report_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/anemia', methods=['GET'])
def predict_anemia():
    global latest_report_data
    try:
        if latest_report_data is None:
            response = requests.get(f"{app.config['BASE_API_URL']}/api/report")
            response.raise_for_status()
            api_data = response.json()
        else:
            api_data = latest_report_data

        input_data = extract_features_anemia(api_data)
        prediction = anemia_model.predict(input_data)[0]
        prediction_proba = anemia_model.predict_proba(input_data)[0]

        result = {
            'prediction': int(prediction),
            'confidence': float(max(prediction_proba)) * 100,
            'anemia_result': "Positive" if prediction_proba[1] > 0.5 else "Negative"
        }
        app.logger.info(f"Anemia prediction completed: {result}")
        return jsonify(result)
    except requests.RequestException as e:
        return jsonify({'error': f"Failed to fetch report data: {str(e)}"}), 500
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

@app.route('/api/predict/dengue', methods=['GET'])
def predict_dengue():
    global latest_report_data
    try:
        if latest_report_data is None:
            response = requests.get(f"{app.config['BASE_API_URL']}/api/report")
            response.raise_for_status()
            api_data = response.json()
        else:
            api_data = latest_report_data

        input_data = extract_features_dengue(api_data)
        probabilities = dengue_model.predict(input_data, verbose=0)
        probability_positive = probabilities[0][0]
        prediction = (probability_positive > 0.5).astype(int)

        confidence = probability_positive if prediction == 1 else 1 - probability_positive
        confidence = float(confidence) * 100

        result = {
            'prediction': int(prediction),
            'confidence': confidence,
            'dengue_result': "Positive" if probability_positive > 0.5 else "Negative"
        }
        app.logger.info(f"Dengue prediction completed: {result}")
        return jsonify(result)
    except requests.RequestException as e:
        return jsonify({'error': f"Failed to fetch report data: {str(e)}"}), 500
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Render provides PORT
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])