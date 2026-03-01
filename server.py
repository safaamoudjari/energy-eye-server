from fastapi import FastAPI, UploadFile, File
import requests
import base64
import os

app = FastAPI()

# Roboflow model endpoint
MODEL_URL = "https://detect.roboflow.com/meter-digits-u17oj/4"

# IMPORTANT: same key name as in Render Environment Variables
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")


@app.get("/")
def home():
    return {"message": "Energy Eye Server Running"}


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # 0) Check API key exists
    if not ROBOFLOW_API_KEY:
        return {
            "error": "ROBOFLOW_API_KEY missing. Add it in Render -> Settings -> Environment Variables.",
            "reading": "",
            "avg_confidence": 0,
            "predictions": []
        }

    # 1) Read image bytes
    image_bytes = await file.read()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    # 2) Call Roboflow with lower confidence to get more detections
    # You can tune confidence/overlap later
    url = f"{MODEL_URL}?api_key={ROBOFLOW_API_KEY}&confidence=0.10&overlap=0.30"

    rf = requests.post(
        url,
        data=encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    # 3) Parse response safely
    try:
        result = rf.json()
    except Exception:
        return {
            "error": "Roboflow did not return JSON.",
            "roboflow_status": rf.status_code,
            "roboflow_text": rf.text,
            "reading": "",
            "avg_confidence": 0,
            "predictions": []
        }

    # 4) If Roboflow returned an error message, return it
    if "error" in result:
        return {
            "error": "Roboflow error",
            "roboflow_status": rf.status_code,
            "roboflow_error": result.get("error"),
            "reading": "",
            "avg_confidence": 0,
            "predictions": result.get("predictions", [])
        }

    predictions = result.get("predictions", [])

    # 5) Sort predictions left-to-right by x coordinate
    predictions_sorted = sorted(predictions, key=lambda p: float(p.get("x", 0)))

    # 6) Build reading from classes that are digits
    digits = []
    confidences = []

    for p in predictions_sorted:
        cls = p.get("class")
        if cls is None:
            continue

        cls_str = str(cls).strip()

        # Keep only digit classes "0".."9"
        if cls_str.isdigit():
            digits.append(cls_str)
            confidences.append(float(p.get("confidence", 0)))

    reading = "".join(digits)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    return {
        "reading": reading,
        "avg_confidence": avg_confidence,
        "num_predictions": len(predictions),
        "roboflow_status": rf.status_code,
        "predictions": predictions
    }

