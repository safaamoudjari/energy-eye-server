from fastapi import FastAPI, Request
import requests
import base64
import os

app = FastAPI()

MODEL_URL = "https://detect.roboflow.com/meter-digits-u17oj/4"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

@app.post("/predict")
async def predict(request: Request):

    # 1) Read raw image مباشرة من ESP32
    image_bytes = await request.body()

    print("📸 Image received, size:", len(image_bytes))

    if not image_bytes:
        return {"error": "Empty image received"}

    # 2) Encode
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    # 3) Call Roboflow
    url = f"{MODEL_URL}?api_key={ROBOFLOW_API_KEY}&confidence=0.10&overlap=0.30"

    rf = requests.post(
        url,
        data=encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    try:
        result = rf.json()
    except:
        return {"error": "Roboflow failed", "text": rf.text}

    predictions = result.get("predictions", [])

    # 4) Sort left → right
    predictions_sorted = sorted(predictions, key=lambda p: float(p.get("x", 0)))

    # 5) Extract digits
    digits = []
    confidences = []

    for p in predictions_sorted:
        cls = str(p.get("class", "")).strip()
        if cls.isdigit():
            digits.append(cls)
            confidences.append(float(p.get("confidence", 0)))

    reading = "".join(digits)
    avg_conf = sum(confidences)/len(confidences) if confidences else 0

    return {
        "reading": reading,
        "avg_confidence": avg_conf,
        "num_predictions": len(predictions)
    }

