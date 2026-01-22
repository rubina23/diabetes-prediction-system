## **10. Web Interface with Gradio**

import gradio as gr
import pandas as pd
import pickle

#Loaded saved model
with open("diabetes_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
    prediction = pipeline.predict(input_data)[0]
    return "Diabetic" if prediction == 1 else "Not Diabetic"

diabetes_app = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="BloodPressure"),
        gr.Number(label="SkinThickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="DiabetesPedigreeFunction"),
        gr.Number(label="Age")
    ],
    outputs="text",
    title="Diabetes Prediction System",
    description="Enter the patient's information to predict diabetes."
)

diabetes_app.launch(share = True)

## **11. Deployment to Hugging Face**
#Hugging Face Spaces public URL: https://huggingface.co/spaces/rubina25/Diabetes-Prediction-System
