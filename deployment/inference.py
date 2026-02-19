import numpy as np
import os
import sys
import torch
from PIL import Image
from io import BytesIO
import urllib.request

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__).replace('/deployment', '')))
from src.config import *
from src.model import get_torch_device, get_transforms, load_checkpoint

# Lazy load model - only load when first needed
_model = None
_model_type = None
_device = get_torch_device()
_transform = None

# Cloud model URL for Streamlit/Hugging Face deployment
DEFAULT_DEPLOY_MODEL_FILENAME = "best_model_resnet.pth"
DEPLOY_MODEL_FILENAME = os.getenv("DEPLOY_MODEL_FILENAME", DEFAULT_DEPLOY_MODEL_FILENAME)
DEFAULT_MODEL_CLOUD_URL = f"https://huggingface.co/abhishekghz/brain-tumor-classifier/resolve/main/{DEPLOY_MODEL_FILENAME}"
MODEL_CLOUD_URL = os.getenv("MODEL_CLOUD_URL", DEFAULT_MODEL_CLOUD_URL)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def _candidate_local_paths():
    candidates = [
        os.path.join(MODEL_DIR, DEPLOY_MODEL_FILENAME),
        os.path.join(MODEL_DIR, MODEL_FILENAME),
        os.path.join(MODEL_DIR, "best_model_resnet.pth"),
        os.path.join(MODEL_DIR, "best_model_vit.pth"),
        os.path.join(MODEL_DIR, "best_model.pth"),
    ]
    seen = set()
    unique_paths = []
    for path in candidates:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)
    return unique_paths


def _resolve_local_model_path():
    for path in _candidate_local_paths():
        if os.path.exists(path):
            return path
    return _candidate_local_paths()[0]


LOCAL_MODEL_PATH = _resolve_local_model_path()

def _download_model(url, save_path):
    """Download model from cloud source."""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    print(f"Downloading model from cloud: {url}")
    try:
        request = urllib.request.Request(url)
        if HF_TOKEN:
            request.add_header("Authorization", f"Bearer {HF_TOKEN}")
        with urllib.request.urlopen(request) as response, open(save_path, "wb") as out_file:
            out_file.write(response.read())
        print(f"Model downloaded successfully")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False


def _candidate_cloud_urls():
    candidates = [MODEL_CLOUD_URL]
    base = "https://huggingface.co/abhishekghz/brain-tumor-classifier/resolve/main"
    for name in [DEPLOY_MODEL_FILENAME, "best_model_resnet.pth", "best_model_vit.pth", "best_model.pth"]:
        url = f"{base}/{name}"
        if url not in candidates:
            candidates.append(url)
    return candidates

def _get_model():
    """Load model lazily on first use."""
    global _model, _model_type, _transform
    if _model is None:
        try:
            local_model_path = _resolve_local_model_path()

            # Try loading from local path first
            if os.path.exists(local_model_path):
                print(f"Loading model from local path: {os.path.basename(local_model_path)}")
                _model, _model_type = load_checkpoint(local_model_path, map_location=_device)
            else:
                # Download from cloud if local doesn't exist (for Streamlit Cloud)
                print("Local model not found. Downloading from cloud...")
                downloaded = False
                for candidate_url in _candidate_cloud_urls():
                    if _download_model(candidate_url, local_model_path):
                        downloaded = True
                        break
                if not downloaded:
                    raise RuntimeError("Could not load model from any configured cloud URL")
                _model, _model_type = load_checkpoint(local_model_path, map_location=_device)
            _model = _model.to(_device)
            _model.eval()
            _transform = get_transforms(_model_type, train=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return _model


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
    
    img = Image.fromarray(img_array)
    _get_model()
    tensor = _transform(img).unsqueeze(0)
    return tensor


def _predict_preprocessed(img):
    model = _get_model()
    img = img.to(_device)
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        confidence, class_id = torch.max(probs, dim=1)
    class_id = int(class_id.item())
    confidence = float(confidence.item())
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
        print(f"Error: Could not decode image from bytes: {e}")
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
