import os

import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATA_DIR,
    IMG_SIZE,
    LABELED_DIR,
    LEARNING_RATE,
    MODEL_DIR,
)


def _load_incremental_data():
    train_dir = os.path.join(DATA_DIR, "Training")

    train_ds = image_dataset_from_directory(
        train_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        shuffle=True,
    )

    incremental_ds = None
    if os.path.isdir(LABELED_DIR):
        labeled_classes = [
            os.path.join(LABELED_DIR, cls)
            for cls in CLASS_NAMES
            if os.path.isdir(os.path.join(LABELED_DIR, cls))
        ]
        if labeled_classes:
            incremental_ds = image_dataset_from_directory(
                LABELED_DIR,
                image_size=(IMG_SIZE, IMG_SIZE),
                batch_size=BATCH_SIZE,
                label_mode="categorical",
                class_names=CLASS_NAMES,
                shuffle=True,
            )

    if incremental_ds is None:
        return train_ds

    return train_ds.concatenate(incremental_ds)


def _load_validation_data():
    val_dir = os.path.join(DATA_DIR, "Testing")
    val_ds = image_dataset_from_directory(
        val_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        shuffle=False,
    )
    return val_ds


def fine_tune_on_new_data(epochs=3):
    model_path = os.path.join(MODEL_DIR, "best_model.keras")
    model = keras.models.load_model(model_path)

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name == "resnet50":
            layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE * 0.1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_ds = _load_incremental_data().prefetch(tf.data.AUTOTUNE)
    val_ds = _load_validation_data().prefetch(tf.data.AUTOTUNE)

    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save(model_path)
    return model
