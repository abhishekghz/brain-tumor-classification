import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.data_loader import load_data
from src.model import build_model, get_torch_device, save_checkpoint
from src.config import *

def train(model_type=MODEL_TYPE, model_filename=MODEL_FILENAME, artifact_suffix=""):
    train_loader, val_loader = load_data(model_type=model_type)
    device = get_torch_device()
    model = build_model(model_type=model_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "accuracy": [],
        "val_accuracy": [],
        "loss": [],
        "val_loss": [],
    }

    for _ in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        val_loss_running = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_running += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_running / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    save_checkpoint(model, model_type=model_type, path=os.path.join(MODEL_DIR, model_filename))

    # Plot accuracy & loss
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(history["accuracy"], label="Train")
    plt.plot(history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history["loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    suffix = f"_{artifact_suffix}" if artifact_suffix else ""
    plt.savefig(os.path.join(RESULTS_DIR, f"accuracy_loss{suffix}.png"))
    plt.close()

    return history
