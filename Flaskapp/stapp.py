import streamlit as st
import os
import google.generativeai as genai
import json
import re
from PIL import Image
import plotly.express as px
import pandas as pd
import pickle

# API Configuration
apikey = os.getenv('medapikey')
genai.configure(api_key=apikey)

# Model Configuration
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

def upload_to_gemini(path, mime_type="image/jpeg"):
    file = genai.upload_file(path, mime_type=mime_type)
    return file

def extract_data_from_image(image_path):
    files = [upload_to_gemini(image_path)]
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message([files[0]])
    
    match = re.search(r"```json\n(.*?)\n```", response.text, re.DOTALL)
    if match:
        json_text = match.group(1)
        return json.loads(json_text)
    else:
        raise ValueError("No valid JSON found in response!")

# Load the pre-trained model
try:
    with open('LRmodel.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'blood_model.pkl' not found. Please ensure it exists in the working directory.")
    loaded_model = None

def predict_outcome(data):
    if loaded_model is None:
        return "Model not available"
    
    # Prepare features for prediction using the exact names the model was trained with
    features = [
        int(data.get('gender', 0)),
        float(data.get('MCHC', 0)),
        float(data.get('MCH', 0)),
        float(data.get('Hemoglobin', 0)),  # This will be renamed to Haemoglobin
        float(data.get('MCV', 0))
    ]
    
    # Use the exact feature names the model expects
    feature_names = ['Gender', 'MCHC', 'MCH', 'Haemoglobin', 'MCV']  # Changed to Haemoglobin
    input_df = pd.DataFrame([features], columns=feature_names)
    
    # Make prediction
    try:
        prediction = loaded_model.predict(input_df)
        return prediction[0]
    except ValueError as e:
        return f"Prediction error: {str(e)}"

# Streamlit Interface
st.title("Blood Report Analyzer with Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload Blood Report Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Blood Report", use_column_width=True)
    
    try:
        data = extract_data_from_image("temp_image.jpg")
        
        # Display raw data
        st.subheader("Extracted Blood Report Data")
        st.json(data)
        
        # Convert data to DataFrame for visualization
        df = pd.DataFrame([data])
        
        # Key Metrics Display
        st.subheader("Selected Blood Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Hemoglobin", f"{data.get('Hemoglobin')} g/dL")
            st.metric("MCV", f"{data.get('MCV')} fL")
            st.metric("Gender", "Male" if data.get('gender') == 1 else "Female")
        
        with col2:
            st.metric("MCH", f"{data.get('MCH')} pg")
            st.metric("MCHC", f"{data.get('MCHC')} g/dL")
        
        # Visualizations
        st.subheader("Blood Parameter Visualizations")
        
        # Bar Chart for selected parameters
        selected_params = {
            'Hemoglobin': float(data.get('Hemoglobin', 0)),
            'MCV': float(data.get('MCV', 0)),
            'MCH': float(data.get('MCH', 0)),
            'MCHC': float(data.get('MCHC', 0))
        }
        fig = px.bar(
            x=list(selected_params.keys()),
            y=list(selected_params.values()),
            title="Selected Blood Parameters",
            labels={'x': 'Parameter', 'y': 'Value'},
            color=list(selected_params.keys())
        )
        st.plotly_chart(fig)

        # Prediction Section
        st.subheader("Health Prediction")
        prediction = predict_outcome(data)
        st.write(f"Model Prediction: {prediction}")
        if isinstance(prediction, (int, float)):
            st.write("Interpretation: " + ("Positive" if prediction > 0 else "Negative"))
        elif isinstance(prediction, str) and prediction.startswith("Prediction error"):
            st.error(prediction)
        else:
            st.write("Prediction format not recognized")

        # Clean up temporary file
        os.remove("temp_image.jpg")
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload a clear image of a blood report
2. Wait for analysis and prediction
3. View selected parameters, chart, and prediction
""")

if __name__ == "__main__":
    pass