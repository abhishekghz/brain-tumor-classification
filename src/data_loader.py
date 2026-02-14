import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.model import get_transforms
from src.config import *

def load_data(model_type=MODEL_TYPE):
    train_dataset = ImageFolder(
        os.path.join(DATA_DIR, "Training"),
        transform=get_transforms(model_type, train=True),
    )
    val_dataset = ImageFolder(
        os.path.join(DATA_DIR, "Testing"),
        transform=get_transforms(model_type, train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader
