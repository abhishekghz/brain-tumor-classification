import numpy as np
import os
import sys
from PIL import Image
from io import BytesIO

import keras

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__).replace('/deployment', '')))
from src.config import *

model = keras.models.load_model(os.path.join(MODEL_DIR, "best_model.keras"))


def _preprocess_image(img):
    """Resize PIL Image to IMG_SIZE and normalize."""
    if isinstance(img, Image.Image):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        img_array = np.array(img)
    else:
        # If it's a numpy array from file reading
        img = Image.fromarray(img)
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        img_array = np.array(img)
    
    # Convert to RGB if grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        # Remove alpha channel if present
        img_array = img_array[:, :, :3]
    
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def _predict_preprocessed(img):
    preds = model.predict(img)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return CLASS_NAMES[class_id], confidence

def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found.")
        return None, None
    
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error: Could not load image from '{img_path}': {e}")
        return None, None
    
    img = _preprocess_image(img)
    return _predict_preprocessed(img)


def predict_image_bytes(img_bytes):
    if not img_bytes:
        print("Error: Empty image payload.")
        return None, None
try:
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        print(f"Error: Could not decode image from bytes: {e}
        print("Error: Could not decode image from bytes")
        return None, None

    img = _preprocess_image(img)
    return _predict_preprocessed(img)

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
