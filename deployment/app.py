from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from src.config import *
import os

app = FastAPI()
model = load_model(os.path.join(MODEL_DIR, "best_model.keras"))

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "prediction": CLASS_NAMES[class_id],
        "confidence": confidence
    }
