import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from src.config import *

def build_vit_model():
    # Use ResNet50 as base model instead of ViT due to compatibility issues
    # This provides similar feature extraction capabilities
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False

    input_layer = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Preprocess input for ResNet50
    x = tf.keras.applications.resnet50.preprocess_input(input_layer)
    
    # Pass through base model
    x = base_model(x, training=False)
    
    # Add classification head
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
