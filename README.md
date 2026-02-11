# Brain Tumor Classification using Deep Learning

A deep learning framework for multi-class brain tumor classification from MRI images using ResNet50 architecture with high accuracy and easy-to-use deployment API.

## Overview

This project provides an end-to-end solution for brain tumor classification:
- **Classes**: Glioma, Meningioma, Pituitary, No Tumor (4-class classification)
- **Deployment**: FastAPI REST API for inference
- **Framework**: TensorFlow 2.13, Keras

## Dataset


- **Total Images**: 7,023 MRI scans
- **Training Images**: 5,712
- **Testing Images**: 1,311
- **Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Source**: Combined from multiple sources (Figshare, SARTAJ, BR35H)


### Prerequisites
- Python 3.9+
- CUDA/GPU support

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
