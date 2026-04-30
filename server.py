from fastapi import FastAPI, File, UploadFile
import httpx
import os

app = FastAPI()

# 🔥 Roboflow model
MODEL_URL = "https://detect.roboflow.com/meter-digits-u17oj/4"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# 🧠 STORE LAST RESULT (for website)
latest_result = {
    "reading": "",
    "avg_confidence": 0,
    "num_predictions": 0
}


# 🌐 Home route
@app.get("/")
def home():
    return {"message": "Server is running 🚀"}


# 🚀 ESP32 sends image here
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    global latest_result

    # 1️⃣ Read image from ESP32
    image_bytes = await file.read()

    if not image_bytes:
        return {"error": "Empty image received"}

    # 2️⃣ Send image to Roboflow AI
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            MODEL_URL,
            params={
                "api_key": ROBOFLOW_API_KEY,
                "confidence": 0.10,
                "overlap": 0.30
            },
            files={"file": ("image.jpg", image_bytes, "image/jpeg")}
        )

    # 3️⃣ Handle API failure
    if response.status_code != 200:
        return {
            "error": "Roboflow failed",
            "text": response.text
        }

    result = response.json()
    predictions = result.get("predictions", [])

    # 4️⃣ Sort digits left → right
    predictions_sorted = sorted(
        predictions,
        key=lambda p: float(p.get("x", 0))
    )

    # 5️⃣ Extract digits + confidence
    digits = [
        str(p["class"])
        for p in predictions_sorted
        if str(p.get("class", "")).isdigit()
    ]

    confidences = [
        float(p.get("confidence", 0))
        for p in predictions_sorted
        if str(p.get("class", "")).isdigit()
    ]

    reading = "".join(digits)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    # 6️⃣ STORE RESULT FOR WEBSITE
    latest_result = {
        "reading": reading,
        "avg_confidence": avg_conf,
        "num_predictions": len(predictions)
    }

    # 7️⃣ RETURN RESULT TO ESP32
    return latest_result


# 🌍 Website reads latest result here
@app.get("/latest")
async def get_latest():
    return latest_result
