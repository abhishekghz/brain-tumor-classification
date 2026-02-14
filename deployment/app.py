from fastapi import FastAPI, UploadFile, File
from deployment.inference import predict_image_bytes

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    label, confidence = predict_image_bytes(contents)
    if label is None:
        return {"prediction": None, "confidence": 0.0, "error": "Invalid image or model unavailable"}

    return {
        "prediction": label,
        "confidence": confidence
    }
