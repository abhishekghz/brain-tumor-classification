import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.config import *

model = load_model(os.path.join(MODEL_DIR, "best_model.h5"))

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))

    return CLASS_NAMES[class_id], confidence

if __name__ == "__main__":
    label, conf = predict_image("sample_mri.jpg")
    print(f"Prediction: {label}, Confidence: {conf:.4f}")
