import os
import google.generativeai as genai
import json
# from fastapi import FastAPI
import re
from flask import Flask, request, jsonify
# from Flaskapp.main import filenamae

apikey="AIzaSyBy3yGbpJvuaMUx_R2sz-Ig3NACOBGzvtY"
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

MCV = data['MCV']


print(f'Data : {data}')
print("hemo:", MCV)

# Haemoglobin = data['Hemoglobin'] or data['Haemoglobin']
# Hematocrit = data['Hematocrit (PCV)']
# RBC_Count = data['RBC Count']
# MCV = data['MCV']
# MCH = data['MCH']
# MCHC = data['MCHC']
# RDW_CV = data['RDW CV']
# RDW_SD = data['RDW SD']
# Total_Leucocyte_Count = data['Total Leucocyte Count']
# NEUTROPHILS = data['Neutrophils']
# LYMPHOCYTES = data['Lymphocytes']
# EOSINOPHILS = data['Eosinophils']
# BASOPHILS = data['Basophils']
# MONOCYTES = data['Monocytes']
# Platelet_Count = data['Platelet Count']
# MPV = data['Mean Platelet Volume (MPV)']
# gender = data['gender']
# Hct = data['Hct']
# PCT= data['PCT']
# PDW= data['PDW']


rep_data=[{
  "MCV": MCV,
#   "Hematocrit": Hematocrit,
#   "RBC_Count": RBC_Count,
}]
# app = FastAPI()
app = Flask(__name__)

@app.get('/report')
def read_report():
  return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)


