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

# Configuration
API_KEY = os.getenv('medapikey')  # Ensure this environment variable is set
IMAGE_PATH = "../rep.jpg"  # Path to the blood report image
API_URL = "http://localhost:5000/report"  # API endpoint for fetching report data

# Configure Google Gemini API
genai.configure(api_key=API_KEY)

# Flask app setup
app = Flask(__name__)
CORS(app)

# Gemini model configuration
GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

SYSTEM_INSTRUCTION = (
    "Extract blood report data and provide observed values in JSON format. "
    "Include gender (1 for male, 0 for female) and age from the report. Normalize 'Hemoglobin' and 'Haemoglobin' to 'Hemoglobin'. "
)

def upload_to_gemini(path, mime_type="image/jpeg"):
    """Upload a file to Gemini and return the file object."""
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        return file
    except Exception as e:
        raise RuntimeError(f"Failed to upload file '{path}': {str(e)}")

def extract_blood_report(image_path):
    """Extract blood report data from an image and return it as a JSON object."""
    try:
        # Define the model within the function
        local_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=GENERATION_CONFIG,
            system_instruction=SYSTEM_INSTRUCTION,
        )

        # Upload the image
        uploaded_file = upload_to_gemini(image_path)

        # Start a chat session with predefined history
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

        # Send extraction request
        response = chat_session.send_message("Extract again")
        result = response.text

        # Extract JSON from response
        match = re.search(r"```json\n(.*?)\n```", result, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON found in response!")
        
        json_text = match.group(1)
        data = json.loads(json_text)

        # Normalize Hemoglobin/Haemoglobin
        if "Haemoglobin" in data:
            data["Hemoglobin"] = data.pop("Haemoglobin")
        
        # Ensure Age is present (case-insensitive)
        if "age" not in data and "Age" not in data:
            data["age"] = None
        elif "Age" in data:
            data["age"] = data.pop("Age")

        return data

    except Exception as e:
        raise RuntimeError(f"Error extracting blood report: {str(e)}")

# Load the trained models
def load_anemia_model(model_path='LRmodel.pkl'):
    """Load the anemia model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load anemia model from '{model_path}': {str(e)}")

def load_dengue_model_and_scaler(model_path='./models/DeNN.pkl', scaler_path='./models/scaler.pkl'):
    """Load the dengue model and scaler from pickle files."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        raise RuntimeError(f"Failed to load dengue model or scaler: {str(e)}")

anemia_model = load_anemia_model()
dengue_model, dengue_scaler = load_dengue_model_and_scaler()

# Feature extraction functions
def extract_features_anemia(api_data):
    """Extract and format features for anemia prediction."""
    try:
        hemoglobin = api_data.get("Hemoglobin", api_data.get("Haemoglobin", 5))  # Default to 5 if missing
        features = {
            'Gender': float(api_data['gender']),
            'Hemoglobin': float(hemoglobin),
            'MCH': float(api_data['MCH']),
            'MCHC': float(api_data['MCHC']),
            'MCV': float(api_data['MCV'])
        }
        return pd.DataFrame([features])
    except KeyError as e:
        raise ValueError(f"Missing required feature: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid data type in API response: {e}")

def extract_features_dengue(api_data):
    """Extract and scale features for dengue prediction."""
    try:
        hemoglobin = api_data.get("Hemoglobin", api_data.get("Haemoglobin"))
        if hemoglobin is None:
            raise ValueError("Hemoglobin or Haemoglobin value is missing")

        # Match the exact feature order from the sample input
        features = [
            float(api_data.get("gender", 1)),  # Gender
            float(api_data.get("age", 0)),     # Age (case-insensitive, default 0)
            float(hemoglobin),                 # Hemoglobin
            float(api_data.get("NEUTROPHILS", api_data.get("Neutrophils", 0))),  # Neutrophils
            float(api_data.get("LYMPHOCYTES", api_data.get("Lymphocytes", 0))),  # Lymphocytes
            float(api_data.get("MONOCYTES", api_data.get("Monocytes", 0))),      # Monocytes
            float(api_data.get("EOSINOPHILS", api_data.get("Eosinophils", 0))),  # Eosinophils
            float(api_data.get("RBC Count", 0)),  # RBC count
            float(api_data.get("Hematocrit (PCV)", api_data.get("Hct", 0))),     # HCT
            float(api_data.get("MCV", 0)),     # MCV
            float(api_data.get("MCH", 0)),     # MCH
            float(api_data.get("MCHC", 0)),    # MCHC
            float(api_data.get("RDW CV", api_data.get("RDW-CV", 0))),           # RDW_CV
            float(api_data.get("Platelet Count", 0)),  # Total Platelet Count
            float(api_data.get("Mean Platelet Volume (MPV)", api_data.get("MPV", 0))),  # MPV
            float(api_data.get("PDW", 0)),     # PDW
            float(api_data.get("PCT", 0)),     # PCT
            float(api_data.get("Total Leucocyte Count", 0))  # Total WBC Count
        ]

        # Convert to NumPy array and reshape
        input_data = np.array(features).reshape(1, -1)
        
        # Scale the input data using the globally loaded scaler
        return dengue_scaler.transform(input_data)

    except KeyError as e:
        raise ValueError(f"Missing required feature: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid data type in API response: {e}")

# API Endpoints
@app.route('/report', methods=['GET'])
def read_report():
    """Return the extracted blood report data."""
    try:
        data = extract_blood_report(IMAGE_PATH)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/anemia', methods=['GET'])
def predict_anemia():
    """Predict anemia based on blood report data."""
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        api_data = response.json()

        input_data = extract_features_anemia(api_data)
        prediction = anemia_model.predict(input_data)[0]
        prediction_proba = anemia_model.predict_proba(input_data)[0]

        result = {
            'prediction': int(prediction),  # 0 = No Anemia, 1 = Anemia
            'confidence': float(max(prediction_proba)) * 100,
            'anemia_result': "Positive" if prediction_proba[1] > 0.5 else "Negative"
        }
        return jsonify(result)

    except requests.RequestException as e:
        return jsonify({'error': f"Failed to fetch API data: {str(e)}"}), 500
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

@app.route('/predict/dengue', methods=['GET'])
def predict_dengue():
    """Predict dengue based on blood report data."""
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        api_data = response.json()

        input_data = extract_features_dengue(api_data)
        probabilities = dengue_model.predict(input_data)  # Keras predict returns [[P(positive)]]
        probability_positive = probabilities[0][0]  # Single output neuron (sigmoid)
        prediction = (probability_positive > 0.5).astype(int)  # Threshold at 0.5

        result = {
            'prediction': int(prediction),  # 0 = No Dengue, 1 = Dengue
            'confidence': float(probability_positive) * 100,  # Probability of positive class
            'dengue_result': "Positive" if probability_positive > 0.5 else "Negative"
        }
        return jsonify(result)

    except requests.RequestException as e:
        return jsonify({'error': f"Failed to fetch API data: {str(e)}"}), 500
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)