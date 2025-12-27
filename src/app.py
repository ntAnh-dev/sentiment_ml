# src/app.py
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import os
import time

MODEL_PATH = "models/best_model.joblib"

app = FastAPI(title="Sentiment ML API")

while not os.path.exists(MODEL_PATH):
    print("Waiting for model to be trained...")
    time.sleep(3)

bundle = joblib.load(MODEL_PATH)
vectorizer = bundle["vectorizer"]
model = bundle["model"]

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: InputText):
    X = vectorizer.transform([input.text])
    pred = model.predict(X)[0]

    return {
        "sentiment": "positive" if pred == 1 else "negative",
        "label": int(pred),
        "model": bundle["model_name"]
    }
