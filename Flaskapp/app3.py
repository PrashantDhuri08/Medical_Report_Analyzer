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
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

# Configuration
API_KEY = os.getenv('medapikey')
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
API_URL = "http://localhost:5000/report"
DB_PATH = './database/medical_reports.db'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('./database', exist_ok=True)

genai.configure(api_key=API_KEY)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
SYSTEM_INSTRUCTION = (
    "Extract blood report data and provide observed values in JSON format. Include gender (1 for male, 0 for female) and age from the report. Normalize 'Hemoglobin' and 'Haemoglobin' to 'Hemoglobin'.Based on the values, provide recommendations for medical actions or lifestyle changes to address abnormalities."
)

# Database functions
def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create reports table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        report_date TIMESTAMP,
        hemoglobin REAL,
        hematocrit REAL,
        rbc_count REAL,
        wbc_count REAL,
        platelet_count REAL,
        mcv REAL,
        mch REAL,
        mchc REAL,
        rdw_cv REAL,
        mpv REAL,
        neutrophils REAL,
        lymphocytes REAL,
        eosinophils REAL,
        monocytes REAL,
        basophils REAL,
        gender INTEGER,
        age INTEGER,
        anemia_risk REAL,
        dengue_risk REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def get_or_create_user(email, name=None):
    """Get a user by email or create a new one if not exists"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT id, name FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    
    if user:
        user_id, user_name = user
        conn.close()
        return user_id, user_name
    
    # Create new user
    cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name or email, email))
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return user_id, name or email

def save_report(user_id, report_data, anemia_risk, dengue_risk):
    """Save report data to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Extract values from report data
    hemoglobin = float(report_data.get('Hemoglobin', 0))
    hematocrit = float(report_data.get('Hematocrit (PCV)', report_data.get('Hct', report_data.get('PCV', 0))))
    rbc_count = float(report_data.get('RBC Count', 0))
    wbc_count = float(report_data.get('Total Leucocyte Count', report_data.get('Total WBC Count', 0)))
    platelet_count = float(report_data.get('Platelet Count', 0))
    mcv = float(report_data.get('MCV', 0))
    mch = float(report_data.get('MCH', 0))
    mchc = float(report_data.get('MCHC', 0))
    rdw_cv = float(report_data.get('RDW CV', report_data.get('RDW-CV', 0)))
    mpv = float(report_data.get('Mean Platelet Volume (MPV)', report_data.get('MPV', 0)))
    neutrophils = float(report_data.get('NEUTROPHILS', report_data.get('Neutrophils', 0)))
    lymphocytes = float(report_data.get('LYMPHOCYTES', report_data.get('Lymphocytes', 0)))
    eosinophils = float(report_data.get('EOSINOPHILS', report_data.get('Eosinophils', 0)))
    monocytes = float(report_data.get('MONOCYTES', report_data.get('Monocytes', 0)))
    basophils = float(report_data.get('BASOPHILS', report_data.get('Basophils', 0)))
    gender = int(report_data.get('gender', 1))
    age = int(report_data.get('age', 0))
    
    # Insert report data
    cursor.execute('''
    INSERT INTO reports (
        user_id, report_date, hemoglobin, hematocrit, rbc_count, wbc_count,
        platelet_count, mcv, mch, mchc, rdw_cv, mpv, neutrophils, lymphocytes,
        eosinophils, monocytes, basophils, gender, age, anemia_risk, dengue_risk
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, datetime.now(), hemoglobin, hematocrit, rbc_count, wbc_count,
        platelet_count, mcv, mch, mchc, rdw_cv, mpv, neutrophils, lymphocytes,
        eosinophils, monocytes, basophils, gender, age, anemia_risk, dengue_risk
    ))
    
    report_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return report_id

def get_user_reports(user_id, limit=10):
    """Get reports for a specific user"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT * FROM reports 
    WHERE user_id = ? 
    ORDER BY report_date DESC 
    LIMIT ?
    ''', (user_id, limit))
    
    reports = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return reports

def get_trend_data(user_id, parameter, limit=5):
    """Get trend data for a specific parameter"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(f'''
    SELECT report_date, {parameter} 
    FROM reports 
    WHERE user_id = ? AND {parameter} IS NOT NULL 
    ORDER BY report_date DESC 
    LIMIT ?
    ''', (user_id, limit))
    
    data = cursor.fetchall()
    conn.close()
    
    # Format data for chart
    dates = [row[0] for row in data]
    values = [row[1] for row in data]
    
    return dates, values

def generate_trend_chart(user_id, parameter, limit=5):
    """Generate a trend chart for a specific parameter"""
    dates, values = get_trend_data(user_id, parameter, limit)
    
    if not dates:
        return None
    
    # Reverse lists to show oldest to newest
    dates.reverse()
    values.reverse()
    
    # Convert dates to datetime objects
    dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in dates]
    
    # Create chart
    plt.figure(figsize=(8, 4))
    plt.plot(dates, values, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.title(f'{parameter.replace("_", " ").title()} Trend')
    plt.xlabel('Date')
    plt.ylabel(parameter.replace("_", " ").title())
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Save chart to bytes
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    # Convert to base64
    chart_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    
    return chart_base64

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_to_gemini(path, mime_type="image/jpeg"):
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        return file
    except Exception as e:
        raise RuntimeError(f"Failed to upload file '{path}': {str(e)}")

def extract_blood_report(image_path):
    try:
        local_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=GENERATION_CONFIG,
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
                  "age": "26",
                  "recommendations": [
                    "Low Hemoglobin may indicate anemia. Increase iron-rich foods or consult a doctor.",
                    "MCV is below normal, suggesting microcytic anemia. Seek medical advice.",
                    "Normal platelet count. Maintain a healthy lifestyle."
                  ]
                }
                ```"""
            ]}
        ])
        response = chat_session.send_message("Extract and provide recommendations")
        result = response.text
        match = re.search(r"```json\n(.*?)\n```", result, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON found in response!")
        json_text = match.group(1)
        data = json.loads(json_text)
        
        # Normalize 'Haemoglobin' to 'Hemoglobin'
        if "Haemoglobin" in data:
            data["Hemoglobin"] = data.pop("Haemoglobin")
        
        # Handle age key normalization
        if "age" not in data and "Age" not in data:
            data["age"] = None
        elif "Age" in data:
            data["age"] = data.pop("Age")
        
        return data
    except Exception as e:
        raise RuntimeError(f"Error extracting blood report: {str(e)}")

def load_anemia_model(model_path='LRmodel.pkl'):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load anemia model from '{model_path}': {str(e)}")

def load_dengue_model_and_scaler(model_path='./models/dengue.keras', scaler_path='./models/scaler.pkl'):
    """Load the dengue TensorFlow model (.keras) and scaler."""
    try:
        # Load TensorFlow model in .keras format
        model = tf.keras.models.load_model(model_path)
        # Load scaler using pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        raise RuntimeError(f"Failed to load dengue model or scaler: {str(e)}")

anemia_model = load_anemia_model()
dengue_model, dengue_scaler = load_dengue_model_and_scaler()

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
    except KeyError as e:
        raise ValueError(f"Missing required feature: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid data type in API response: {e}")

def extract_features_dengue(api_data):
    try:
        hemoglobin = api_data.get("Hemoglobin", api_data.get("Haemoglobin"))
        if hemoglobin is None:
            raise ValueError("Hemoglobin or Haemoglobin value is missing")
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
    except KeyError as e:
        raise ValueError(f"Missing required feature: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid data type in API response: {e}")

latest_report_data = None

@app.route('/upload', methods=['POST'])
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

        # Get user information from request
        user_email = request.form.get('email', 'anonymous@example.com')
        user_name = request.form.get('name', None)

        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # Extract report data
        report_data = extract_blood_report(image_path)
        latest_report_data = report_data
        
        # Get or create user
        user_id, user_name = get_or_create_user(user_email, user_name)
        
        # Calculate risks
        try:
            # Anemia risk
            input_data = extract_features_anemia(report_data)
            anemia_prediction = anemia_model.predict(input_data)[0]
            anemia_confidence = float(max(anemia_model.predict_proba(input_data)[0])) * 100
            
            # Dengue risk
            input_data = extract_features_dengue(report_data)
            dengue_probabilities = dengue_model.predict(input_data)
            dengue_probability = dengue_probabilities[0][0]
            dengue_prediction = (dengue_probability > 0.5).astype(int)
            dengue_confidence = dengue_probability if dengue_prediction == 1 else 1 - dengue_probability
            dengue_confidence = float(dengue_confidence) * 100
            
            # Save to database
            save_report(user_id, report_data, anemia_confidence, dengue_confidence)
            
            # Clean up
            os.remove(image_path)
            
            return jsonify({
                'message': 'Image uploaded and processed successfully', 
                'data': report_data,
                'user_id': user_id,
                'anemia_risk': {
                    'prediction': int(anemia_prediction),
                    'confidence': anemia_confidence
                },
                'dengue_risk': {
                    'prediction': int(dengue_prediction),
                    'confidence': dengue_confidence
                }
            })
        except Exception as e:
            return jsonify({'error': f'Error processing report: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/report', methods=['GET'])
def get_report():
    global latest_report_data
    try:
        if latest_report_data is None:
            return jsonify({'error': 'No report data available. Please upload an image first.'}), 400
        return jsonify(latest_report_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/anemia', methods=['GET'])
def predict_anemia():
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        api_data = response.json()

        input_data = extract_features_anemia(api_data)
        prediction = anemia_model.predict(input_data)[0]
        prediction_proba = anemia_model.predict_proba(input_data)[0]

        result = {
            'prediction': int(prediction),
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
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        api_data = response.json()

        input_data = extract_features_dengue(api_data)
        probabilities = dengue_model.predict(input_data)
        probability_positive = probabilities[0][0]
        prediction = (probability_positive > 0.5).astype(int)

        # Adjust confidence to reflect the predicted class
        confidence = probability_positive if prediction == 1 else 1 - probability_positive
        confidence = float(confidence) * 100

        result = {
            'prediction': int(prediction),
            'confidence': confidence,
            'dengue_result': "Positive" if probability_positive > 0.5 else "Negative"
        }
        return jsonify(result)
    except requests.RequestException as e:
        return jsonify({'error': f"Failed to fetch API data: {str(e)}"}), 500
    
@app.route('/user/<int:user_id>/reports', methods=['GET'])
def get_user_reports_api(user_id):
    """API endpoint to get reports for a specific user"""
    try:
        limit = request.args.get('limit', 10, type=int)
        reports = get_user_reports(user_id, limit)
        return jsonify({'reports': reports})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user/<int:user_id>/trend/<parameter>', methods=['GET'])
def get_trend_chart_api(user_id, parameter):
    """API endpoint to get trend chart for a specific parameter"""
    try:
        limit = request.args.get('limit', 5, type=int)
        chart_base64 = generate_trend_chart(user_id, parameter, limit)
        
        if chart_base64:
            return jsonify({'chart': chart_base64})
        else:
            return jsonify({'error': 'No data available for trend analysis'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user/<int:user_id>/trends', methods=['GET'])
def get_all_trends_api(user_id):
    """API endpoint to get trend charts for all parameters"""
    try:
        parameters = [
            'hemoglobin', 'hematocrit', 'rbc_count', 'wbc_count', 
            'platelet_count', 'mcv', 'mch', 'mchc'
        ]
        
        trends = {}
        for param in parameters:
            chart_base64 = generate_trend_chart(user_id, param)
            if chart_base64:
                trends[param] = chart_base64
        
        return jsonify({'trends': trends})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize database
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)