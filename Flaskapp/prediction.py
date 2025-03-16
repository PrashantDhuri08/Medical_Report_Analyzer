from flask import Flask, jsonify
import pickle
import pandas as pd
import requests

app = Flask(__name__)

# Load the trained model (e.g., Random Forest or Logistic Regression)
with open('LRmodel.pkl', 'rb') as f:  # Use 'LRmodel.pkl' if you prefer Logistic Regression
    model = pickle.load(f)

# API endpoint to fetch the data (replace with your actual API URL)
API_URL = "http://localhost:5000/report"  # Replace with the actual API URL

def extract_features(api_data):
    """Extract relevant features from API response and format for model input."""
    try:
        # Extract only the required features
        features = {
            'Gender': float(api_data['gender']),
            'Hemoglobin': float(api_data['Haemoglobin']),
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
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]  # Probability scores

        # Prepare response
        result = {
            'prediction': int(prediction),  # 0 = No Anemia, 1 = Anemia
            'confidence': float(max(prediction_proba)) * 100,  # Confidence in percentage
            'input_data': api_data  # Include original data for reference
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