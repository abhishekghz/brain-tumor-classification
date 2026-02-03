# Brain Tumor Classification using Vision Transformer (ViT)

## Overview
This project presents a transformer-based deep learning framework for
multi-class brain tumor classification using MRI images.

Classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor

## Dataset
Combination of:
- Figshare Brain MRI
- SARTAJ Dataset
- BR35H Dataset

Total Images: 7023

## Model
- Vision Transformer (ViT-B/16)
- Transfer learning
- Grad-CAM explainability

## Project Structure
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/

models/best_model.h5
results/
├── accuracy_loss.png
├── confusion_matrix.png
├── classification_report.txt
