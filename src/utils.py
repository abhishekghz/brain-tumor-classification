import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def normalize_img(img):
    return img / 255.0

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
