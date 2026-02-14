# Brain Tumor Classification using Deep Learning

A deep learning framework for multi-class brain tumor classification from MRI images with both a Vision Transformer (ViT) and ResNet50 baseline for research-grade comparison and deployment.

## Overview

This project provides an end-to-end solution for brain tumor classification:
- **Classes**: Glioma, Meningioma, Pituitary, No Tumor (4-class classification)
- **Deployment**: FastAPI REST API for inference
- **Framework**: PyTorch

## 🚀 Live Demo

**Try the app now:** [https://brain-tumor-classification-7100.streamlit.app](https://brain-tumor-classification-7100.streamlit.app)

Features:
- Upload MRI images and get instant predictions
- Admin panel for labeling new data
- Continual learning: Model improves with new labeled data
- Medical-friendly interface for healthcare professionals

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

You can switch architecture from [src/config.py](src/config.py):
- `MODEL_TYPE = "vit"` for Vision Transformer
- `MODEL_TYPE = "resnet"` for ResNet50 baseline

### Vision Transformer (ViT)
- Patch embedding using `Conv2D` patch projection
- Learned positional embedding
- Multi-head self-attention encoder blocks
- MLP head + softmax classifier

### ResNet50 Baseline
- Pre-trained on ImageNet
- Frozen feature extractor + custom classification head

### Run comparison for paper
1. Set `MODEL_TYPE = "resnet"` in [src/config.py](src/config.py), then run training/evaluation.
2. Set `MODEL_TYPE = "vit"` in [src/config.py](src/config.py), then run training/evaluation.
3. Compare metrics from [results/classification_report.txt](results/classification_report.txt) and [results/confusion_matrix.png](results/confusion_matrix.png).

Automated alternative:
- Run `python run_experiments.py` to train/evaluate both models and generate:
  - `results/model_comparison.csv`
  - `results/model_comparison.md`
  - per-model reports/plots (suffix `_resnet` / `_vit`)



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
