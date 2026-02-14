import os
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATA_DIR,
    IMG_SIZE,
    LABELED_DIR,
    LEARNING_RATE,
    MODEL_FILENAME,
    MODEL_DIR,
    MODEL_TYPE,
)
from src.model import get_torch_device, get_transforms, load_checkpoint, save_checkpoint


def _load_incremental_data():
    train_dir = os.path.join(DATA_DIR, "Training")

    train_ds = ImageFolder(
        train_dir,
        transform=get_transforms(MODEL_TYPE, train=True),
    )

    incremental_ds = None
    if os.path.isdir(LABELED_DIR):
        labeled_classes = [
            os.path.join(LABELED_DIR, cls)
            for cls in CLASS_NAMES
            if os.path.isdir(os.path.join(LABELED_DIR, cls))
        ]
        if labeled_classes:
            incremental_ds = ImageFolder(
                LABELED_DIR,
                transform=get_transforms(MODEL_TYPE, train=True),
            )

    if incremental_ds is None:
        return train_ds

    return ConcatDataset([train_ds, incremental_ds])


def _load_validation_data():
    val_dir = os.path.join(DATA_DIR, "Testing")
    val_ds = ImageFolder(
        val_dir,
        transform=get_transforms(MODEL_TYPE, train=False),
    )
    return val_ds


def fine_tune_on_new_data(epochs=3):
    device = get_torch_device()
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    model, model_type = load_checkpoint(model_path, map_location=device)
    model = model.to(device)

    train_ds = _load_incremental_data()
    val_ds = _load_validation_data()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                _ = model(images)

    save_checkpoint(model, model_type=model_type, path=model_path)
    return model
