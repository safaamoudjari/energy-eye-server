from fastapi import FastAPI, File, UploadFile
import httpx
import os

app = FastAPI()

MODEL_URL = "https://detect.roboflow.com/meter-digits-u17oj/4"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")


@app.get("/")
def home():
    return {"message": "Server is running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # 1️⃣ قراءة الصورة
    image_bytes = await file.read()

    if not image_bytes:
        return {"error": "Empty image received"}

    # 2️⃣ إرسال إلى Roboflow
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

    if response.status_code != 200:
        return {
            "error": "Roboflow failed",
            "text": response.text
        }

    result = response.json()
    predictions = result.get("predictions", [])

    # 3️⃣ ترتيب الأرقام من اليسار لليمين
    predictions_sorted = sorted(predictions, key=lambda p: float(p.get("x", 0)))

    # 4️⃣ استخراج الأرقام
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
    avg_conf = sum(confidences)/len(confidences) if confidences else 0

    return {
        "reading": reading,
        "avg_confidence": avg_conf,
        "num_predictions": len(predictions)
    }

