
from fastapi import FastAPI, UploadFile, File
import requests
import base64
import os

app = FastAPI()

MODEL_URL = "https://detect.roboflow.com/meter-digits-u17oj/4"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

@app.get("/")
def home():
    return {"message": "Energy Eye Server Running "}

@app.get("/health")
def health():
    return {"status": "OK "}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    response = requests.post(
        f"{MODEL_URL}?api_key={ROBOFLOW_API_KEY}",
        data=encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    result = response.json()
    predictions = result.get("predictions", [])

    # ترتيب الأرقام من اليسار لليمين
    predictions_sorted = sorted(predictions, key=lambda p: float(p.get("x", 0)))

    digits = []
    confidences = []

    for p in predictions_sorted:
        cls = p.get("class")
        if cls is None:
            continue

        cls_str = str(cls).strip()

        if cls_str.isdigit():
            digits.append(cls_str)
            confidences.append(float(p.get("confidence", 0)))

    reading = "".join(digits)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    return {
        "reading": reading,
        "avg_confidence": avg_confidence,
        "predictions": predictions
    }
