import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
from src.config import *

def _resnet_weights(pretrained):
    return ResNet50_Weights.IMAGENET1K_V2 if pretrained else None


def _vit_weights(pretrained):
    return ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None


def build_resnet_model(pretrained=PRETRAINED):
    model = models.resnet50(weights=_resnet_weights(pretrained))
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model


def build_vit_model(pretrained=PRETRAINED):
    model = models.vit_b_16(weights=_vit_weights(pretrained))
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, NUM_CLASSES)
    return model


def build_model(model_type=MODEL_TYPE, pretrained=PRETRAINED):
    model_type = model_type.lower().strip()
    if model_type == "resnet":
        return build_resnet_model(pretrained=pretrained)
    if model_type == "vit":
        return build_vit_model(pretrained=pretrained)
    raise ValueError(
        f"Unsupported MODEL_TYPE='{model_type}'. Use one of: {SUPPORTED_MODELS}."
    )


def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms(model_type=MODEL_TYPE, train=False):
    model_type = model_type.lower().strip()
    weights = _vit_weights(PRETRAINED) if model_type == "vit" else _resnet_weights(PRETRAINED)

    if weights is not None:
        base_transform = weights.transforms()
        if train:
            return transforms.Compose(
                [
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    base_transform,
                ]
            )
        return base_transform

    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def save_checkpoint(model, model_type=MODEL_TYPE, path=None):
    checkpoint_path = path or os.path.join(MODEL_DIR, MODEL_FILENAME)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": model_type,
            "class_names": CLASS_NAMES,
        },
        checkpoint_path,
    )


def load_checkpoint(path=None, map_location=None):
    checkpoint_path = path or os.path.join(MODEL_DIR, MODEL_FILENAME)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_type = checkpoint.get("model_type", MODEL_TYPE)
    model = build_model(model_type=model_type, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model, model_type
