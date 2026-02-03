import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Image parameters
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
NUM_CLASSES = 4

# Classes
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Training
LEARNING_RATE = 1e-4

# Model
MODEL_NAME = "google/vit-base-patch16-224"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
