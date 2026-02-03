import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.config import *

def generate_gradcam(img_path):
    model = load_model(os.path.join(MODEL_DIR, "best_model.h5"))

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img_tensor = np.expand_dims(img, axis=0)

    preds = model(img_tensor)
    class_idx = tf.argmax(preds[0])

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.output]
    )

    with tf.GradientTape() as tape:
        outputs = grad_model(img_tensor)
        loss = outputs[:, class_idx]

    grads = tape.gradient(loss, img_tensor)
    heatmap = tf.reduce_mean(grads, axis=-1)[0]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    heatmap = cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(heatmap, 0.4, img.astype("uint8"), 0.6, 0)
    return superimposed
