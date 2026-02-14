import os
import pandas as pd

from src.config import RESULTS_DIR, SUPPORTED_MODELS
from src.train import train
from src.evaluate import evaluate


def run_all_experiments(models=None):
    selected_models = models or SUPPORTED_MODELS
    rows = []

    for model_type in selected_models:
        model_filename = f"best_model_{model_type}.pth"
        train(model_type=model_type, model_filename=model_filename, artifact_suffix=model_type)
        metrics = evaluate(model_type=model_type, model_filename=model_filename, artifact_suffix=model_type)
        rows.append(
            {
                "model": model_type,
                "accuracy": metrics["accuracy"],
                "checkpoint": model_filename,
            }
        )

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    md_path = os.path.join(RESULTS_DIR, "model_comparison.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as file:
        file.write(df.to_markdown(index=False))
    return df


if __name__ == "__main__":
    summary = run_all_experiments()
    print("Experiment summary:")
    print(summary.to_string(index=False))
