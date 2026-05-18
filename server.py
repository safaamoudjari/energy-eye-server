from fastapi import FastAPI, Request
import httpx
import os
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or replace * with your exact website URL
    allow_methods=["*"],
    allow_headers=["*"],
)
#  Roboflow model
MODEL_URL = "https://detect.roboflow.com/meter-digits-u17oj/4"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

#  STORE LAST RESULT (for website)
latest_result = {
    "reading": "",
    "avg_confidence": 0,
    "num_predictions": 0
}


#  Home route
@app.get("/")
def home():
    return {"message": "Server is running "}


#  ESP32 sends image here
@app.post("/predict")
async def predict(request: Request):

    global latest_result

    image_bytes = await request.body()

    if not image_bytes:
        return {"error": "Empty image"}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            MODEL_URL,
            params={
                "api_key": ROBOFLOW_API_KEY,
                "confidence": 0.10,
                "overlap": 0.30
            },
            content=image_bytes,
            headers={"Content-Type": "image/jpeg"}
        )

    result = response.json()

    predictions = result.get("predictions", [])

    predictions_sorted = sorted(
        predictions,
        key=lambda p: float(p.get("x", 0))
    )

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

    avg_conf = (
        sum(confidences) / len(confidences)
        if confidences else 0
    )

    latest_result = {
        "reading": reading,
        "avg_confidence": avg_conf,
        "num_predictions": len(predictions)
    }

    return latest_result


#  Website reads latest result here
@app.get("/latest")
async def get_latest():
    return latest_result
