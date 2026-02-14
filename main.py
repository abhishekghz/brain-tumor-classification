from src.train import train
from src.evaluate import evaluate
from src.config import MODEL_TYPE, MODEL_FILENAME

if __name__ == "__main__":
    train(model_type=MODEL_TYPE, model_filename=MODEL_FILENAME, artifact_suffix=MODEL_TYPE)
    evaluate(model_type=MODEL_TYPE, model_filename=MODEL_FILENAME, artifact_suffix=MODEL_TYPE)
