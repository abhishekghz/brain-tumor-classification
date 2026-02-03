import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.model import build_vit_model
from src.config import *

def train():
    train_ds, val_ds = load_data()
    model = build_vit_model()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    model.save(os.path.join(MODEL_DIR, "best_model.keras"))

    # Plot accuracy & loss
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_loss.png"))
    plt.close()
