# Brain Tumor Classification using Deep Learning

A deep learning framework for multi-class brain tumor classification from MRI images using ResNet50 architecture with high accuracy and easy-to-use deployment API.

## Overview

This project provides an end-to-end solution for brain tumor classification:
- **Model**: ResNet50 with transfer learning
- **Accuracy**: 94.49% training, 93.21% validation
- **Classes**: Glioma, Meningioma, Pituitary, No Tumor (4-class classification)
- **Deployment**: FastAPI REST API for inference
- **Framework**: TensorFlow 2.13, Keras

## Dataset

- **Total Images**: 7,023 MRI scans
- **Training Images**: 5,712
- **Testing Images**: 1,311
- **Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Source**: Combined from multiple sources (Figshare, SARTAJ, BR35H)

## Project Structure

```
brain-tumor-classification/
├── data/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── src/
│   ├── config.py          # Configuration and constants
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── model.py           # ResNet50 model architecture
│   ├── train.py           # Training script
│   ├── evaluate.py        # Model evaluation
│   ├── gradcam.py         # Grad-CAM visualization
│   └── utils.py           # Utility functions
├── deployment/
│   ├── app.py             # FastAPI server
│   └── inference.py       # Inference script
├── notebooks/
│   └── EDA_and_visualization.ipynb  # Data exploration
├── models/
│   └── best_model.keras   # Trained model (100+ MB, generated locally)
├── results/
│   ├── accuracy_loss.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
├── main.py                # Main entry point
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

### Prerequisites
- Python 3.9+
- CUDA/GPU support (optional but recommended)
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/abhishekghz/brain-tumor-classification.git
cd brain-tumor-classification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Model Architecture

**ResNet50** (Residual Network with 50 layers)
- Pre-trained on ImageNet
- Transfer learning approach
- Fine-tuned for brain tumor classification
- Frozen base layers + custom classification head:
  - Global Average Pooling 2D
  - Dense layer (512 units, ReLU activation)
  - Dropout (0.4)
  - Output layer (4 units, Softmax activation)

### Performance Metrics

| Metric | Value |
|--------|-------|
| Training Accuracy | 94.49% |
| Validation Accuracy | 93.21% |
| Training Loss | 0.1488 |
| Validation Loss | 0.1903 |
| Model Size | ~100 MB |

## Usage

### 1. Training the Model

```bash
python main.py
```

This will:
- Load and preprocess the training/validation data
- Build the ResNet50 model
- Train for 25 epochs
- Save the best model as `models/best_model.keras`
- Generate evaluation metrics and visualizations

### 2. Running Inference

**Option A: Using the inference script**
```bash
# Simple test
python -c "from deployment.inference import predict_image; print(predict_image('data/Testing/glioma/Te-gl_0284.jpg'))"
```

**Option B: Using the FastAPI server**

Terminal 1 - Start the server:
```bash
pip install python-multipart
uvicorn deployment.app:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 - Make predictions:
```bash
# Using curl
curl -X POST "http://localhost:8000/predict/" \
  -F "file=@data/Testing/glioma/Te-gl_0284.jpg"

# Using Python requests
import requests

with open('data/Testing/glioma/Te-gl_0284.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict/', files=files)
    print(response.json())
```

### 3. Model Evaluation

Generate classification report and confusion matrix:
```bash
python -c "from src.evaluate import evaluate; evaluate()"
```

## API Endpoints

### FastAPI Server

**Base URL**: `http://localhost:8000`

**Prediction Endpoint**
- **Method**: POST
- **Path**: `/predict/`
- **Input**: Image file (JPEG/PNG)
- **Output**: 
```json
{
    "prediction": "glioma",
    "confidence": 0.63
}
```

**Example Response**:
```json
{
    "prediction": "glioma",
    "confidence": 0.6300
}
```

## Configuration

Edit `src/config.py` to modify:
- Image size (default: 224x224)
- Learning rate (default: 0.001)
- Batch size (default: 32)
- Number of epochs (default: 25)
- Model directory path
- Class names

```python
IMG_SIZE = 224
LEARNING_RATE = 0.001
EPOCHS = 25
BATCH_SIZE = 32
NUM_CLASSES = 4
MODEL_DIR = "models"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
```

## Data Pipeline

1. **Data Loading** (`src/data_loader.py`)
   - Load images from directory structure
   - Resize to 224x224 pixels
   - Normalize pixel values to [0, 1]
   - Create train/validation/test splits

2. **Preprocessing**
   - ResNet50 input preprocessing
   - Mean normalization
   - Batch creation

3. **Augmentation** (built into Keras ImageDataGenerator)
   - Rotation
   - Width/height shift
   - Zoom


## Citation

If you use this project in your research, please cite:
```bibtex
@software{gautam2026braintumor,
  author = {Gautam, Abhishek},
  title = {Brain Tumor Classification using Deep Learning},
  year = {2026},
  url = {https://github.com/abhishekghz/brain-tumor-classification}
}
```

## Contact

For questions or suggestions, please open an issue on [GitHub Issues](https://github.com/abhishekghz/brain-tumor-classification/issues).

---

**Last Updated**: February 4, 2026
**Project Status**: ✅ Complete and Production Ready
