import click
import pandas as pd

from pathlib import Path
from joblib import load
from sklearn.pipeline import Pipeline

@click.command()
@click.option(
    "-d",
    "--test-dataset-path",
    default="data/test.csv",
    type=click.Path(exists=True,
                    dir_okay=False,
                    path_type=Path),
    show_default=True,
    help="Path to the test dataset file."
)
@click.option(
    "-s",
    "--sample-submission-path",
    default="data/sampleSubmission.csv",
    type=click.Path(exists=True,
                    dir_okay=False,
                    path_type=Path),
    show_default=True,
    help="Path to the sample submission file."
)
@click.option(
    "-m",
    "--model-path",
    default="model/model.joblib",
    type=click.Path(exists=True,
                    dir_okay=False,
                    path_type=Path),
    show_default=True,
    help="Path to the model."
)
@click.option(
    "--save-predictions-path",
    default="data/predictions.csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
    help="Path to save the predictions file."
)
def predict(
    test_dataset_path : Path,
    sample_submission_path : Path,
    model_path : Path,
    save_predictions_path : Path
) -> None:
    X_test = pd.read_csv(test_dataset_path, index_col="Id")
    model = load(model_path)
    predictions = model.predict(X_test)
    submission = pd.read_csv(sample_submission_path)
    submission["Cover_Type"] = predictions
    #saving predcitions
    save_predictions_path.unlink(missing_ok=True)
    submission.to_csv(save_predictions_path, index=False)
    click.echo(f"Predictions is saved to {save_predictions_path}.")