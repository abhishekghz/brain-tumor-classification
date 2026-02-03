import cv2
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__).replace('/deployment', '')))
from src.config import *

model = load_model(os.path.join(MODEL_DIR, "best_model.keras"))

def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found.")
        return None, None
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from '{img_path}'")
        return None, None
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))

    return CLASS_NAMES[class_id], confidence

if __name__ == "__main__":
    # Example usage with a test image from the data directory
    test_image_path = "data/Testing/glioma/Te-gl_0.jpg"
    
    if os.path.exists(test_image_path):
        label, conf = predict_image(test_image_path)
        if label:
            print(f"Prediction: {label}, Confidence: {conf:.4f}")
    else:
        print(f"To test inference, provide a valid image path.")
        print(f"Usage: python -c \"from deployment.inference import predict_image; print(predict_image('path/to/image.jpg'))\"")
        print(f"\nExample test image paths available:")
        for class_name in CLASS_NAMES:
            class_path = f"data/Testing/{class_name}"
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
                if images:
                    print(f"  - {class_path}/{images[0]}")
