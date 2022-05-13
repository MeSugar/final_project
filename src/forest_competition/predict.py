import click
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline

def predict(
    test_dataset_path : Path,
    sample_submission_path : Path,
    save_predictions_path : Path,
    pipeline : Pipeline
) -> None:
    X_test = pd.read_csv(test_dataset_path, index_col="Id")
    predictions = pipeline.predict(X_test)
    submission = pd.read_csv(sample_submission_path)
    submission["Cover_Type"] = predictions
    #saving predcitions
    path_folder = save_predictions_path.parent
    path_folder.mkdir(exist_ok=True)
    save_predictions_path.unlink(missing_ok=True)
    submission.to_csv(save_predictions_path, index=False)
    click.echo(f"Predictions is saved to {save_predictions_path}.")