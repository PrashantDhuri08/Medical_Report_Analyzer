
import os
import google.generativeai as genai
import json
# from fastapi import FastAPI
from flask_cors import CORS
import re
from flask import Flask, request, jsonify
import requests
import pickle
import pandas as pd
# from Flaskapp.main import filenamae

apikey = os.getenv('medapikey')
image= "../rep.jpg"



genai.configure(api_key=apikey)

def upload_to_gemini(path, mime_type=None):

  file = genai.upload_file(image, mime_type=mime_type)
  # print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash",
  generation_config=generation_config,
  system_instruction="extracts blood report and give observed values in form of json and gender also (if male = 1 &female =0 ) \n\n",
)

# TODO Make these files available on the local file system
# You may need to update the file paths
files = [
  upload_to_gemini("", mime_type="image/jpeg"),
  # upload_to_gemini("Image February 16, 2025 - 3:47PM.jpeg", mime_type="image/jpeg"),
]

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        files[0],
      ],
    },
    {
      "role": "model",
      "parts": [
        "```json\n{\n  \"Hemoglobin\": \"9.0\",\n  \"Hematocrit (PCV)\": \"27\",\n  \"RBC Count\": \"6.1\",\n  \"MCV\": \"44\",\n  \"MCH\": \"15\",\n  \"MCHC\": \"33\",\n  \"RDW CV\": \"20\",\n  \"RDW SD\": \"37.4\",\n  \"Total Leucocyte Count\": \"8300\",\n  \"NEUTROPHILS\": \"78\",\n  \"LYMPHOCYTES\": \"16\",\n  \"EOSINOPHILS\": \"04\",\n  \"BASOPHILS\": \"00\",\n  \"MONOCYTES\": \"02\",\n  \"Platelet Count\": \"225000\",\n  \"Mean Platelet Volume (MPV)\": \"8\"\n}\n```",
      ],
    },
    {
      "role": "user",
      "parts": [
        # files[1],
        "extract \n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "```json\n{\n  \"Hemoglobin\": \"9.0\",\n  \"Hematocrit (PCV)\": \"27\",\n  \"RBC Count\": \"6.1\",\n  \"MCV\": \"44\",\n  \"MCH\": \"15\",\n  \"MCHC\": \"33\",\n  \"RDW CV\": \"20\",\n  \"RDW SD\": \"37.4\",\n  \"Total Leucocyte Count\": \"8300\",\n  \"NEUTROPHILS\": \"78\",\n  \"LYMPHOCYTES\": \"16\",\n  \"EOSINOPHILS\": \"04\",\n  \"BASOPHILS\": \"00\",\n  \"MONOCYTES\": \"02\",\n  \"Platelet Count\": \"225000\",\n  \"Mean Platelet Volume (MPV)\": \"8\",\n  \"gender\": 1\n}\n```",
      ],
    },
  ]
)

response = chat_session.send_message("Extract again")

result= response.text


# print(result)


# Extract JSON from the response
match = re.search(r"```json\n(.*?)\n```", result, re.DOTALL)
if match:
    json_text = match.group(1)
else:
    raise ValueError("No valid JSON found in response!")

data = json.loads(json_text)




app = Flask(__name__)

CORS(app)

@app.get('/report')
def read_report():
  return jsonify(data)


## Load the trained model *********************************************
with open('LRmodel.pkl', 'rb') as f:  
    anemodel = pickle.load(f)



# API endpoint to fetch the data
API_URL = "http://localhost:5000/report"  

def extract_features(api_data):
    """Extract relevant features from API response and format for model input."""
    try:
        # Extract only the required features
        features = {
            'Gender': float(api_data['gender']),
            'Hemoglobin': 5,
            'MCH': float(api_data['MCH']),
            'MCHC': float(api_data['MCHC']),
            'MCV': float(api_data['MCV'])
        }
        # Convert to DataFrame with a single row
        input_df = pd.DataFrame([features])
        return input_df
    except KeyError as e:
        raise ValueError(f"Missing required feature: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid data type in API response: {e}")

@app.route('/predict', methods=['GET'])
def predict_anemia():
    try:
        # Fetch data from the API
        response = requests.get(API_URL)
        response.raise_for_status()  # Raise an error for bad status codes
        api_data = response.json()

        # Extract and preprocess features
        input_data = extract_features(api_data)

        # Make prediction
        prediction = anemodel.predict(input_data)[0]
        prediction_proba = anemodel.predict_proba(input_data)[0]  # Probability scores

        # Prepare response
        result = {
            'prediction': int(prediction),  # 0 = No Anemia, 1 = Anemia
            'confidence': float(max(prediction_proba)) * 100,  # Confidence in percentage
            # 'input_data': api_data  
        }
        return jsonify(result)

    except requests.RequestException as e:
        return jsonify({'error': f"Failed to fetch API data: {str(e)}"}), 500
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)


