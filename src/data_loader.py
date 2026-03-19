import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.model import get_transforms
from src.config import *

def load_data(model_type=MODEL_TYPE):
    use_cuda = torch.cuda.is_available()
    worker_count = 2 if use_cuda else 0

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
        num_workers=worker_count,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=use_cuda,
    )

    return train_loader, val_loader
