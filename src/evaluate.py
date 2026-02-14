import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.model import get_torch_device, load_checkpoint
from src.config import *

def evaluate(model_type=MODEL_TYPE, model_filename=MODEL_FILENAME, artifact_suffix=""):
    _, test_loader = load_data(model_type=model_type)
    device = get_torch_device()
    model, _ = load_checkpoint(os.path.join(MODEL_DIR, model_filename), map_location=device)
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    suffix = f"_{artifact_suffix}" if artifact_suffix else ""
    with open(os.path.join(RESULTS_DIR, f"classification_report{suffix}.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix{suffix}.png"))
    plt.close()

    return {
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "report": report,
    }
