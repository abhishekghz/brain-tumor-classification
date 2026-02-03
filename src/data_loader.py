import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from src.config import *

def load_data():
    train_ds = image_dataset_from_directory(
        os.path.join(DATA_DIR, "Training"),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True
    )

    val_ds = image_dataset_from_directory(
        os.path.join(DATA_DIR, "Testing"),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds
