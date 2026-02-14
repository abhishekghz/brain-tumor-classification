import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_img(img):
    return img / 255.0

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
