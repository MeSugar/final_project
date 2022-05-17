import pip
import click
import mlflow
import mlflow.sklearn

import pandas as pd

from pathlib import Path
from joblib import dump

from .data import get_data
from .pipeline import build_pipeline
from .model import init_classifier, model_evaluation, model_tuning
from .predict import predict


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True,
                    dir_okay=False,
                    path_type=Path),
    show_default=True,
    help="Path to the dataset file."
)
@click.option(
    "-s",
    "--save-model-path",
    default="model/model.joblib",
    type=click.Path(
        dir_okay=False,
        writable=True,
        path_type=Path),
    show_default=True,
    help="Path where the fitted model will be saved."
)
@click.option(
    "--reduce-dim",
    default=None,
    type=click.Choice(["pca", "boruta"]),
    show_default=True,
    help="Method to reduce feature dimensionality."
)
@click.option(
    "--classifier",
    default="knn",
    type=click.Choice(["knn", "logreg", "rfc"]),
    show_default=True,
    help="Algorithm to be used for modeling."
)
def train(
        dataset_path: Path,
        save_model_path: Path,
        reduce_dim : str,
        classifier : str
) -> None:
    """Script to train and save model."""
    X, y = get_data(
        dataset_path
    )
    clf = init_classifier(classifier)
    pipeline = build_pipeline(
        reduce_dim,
        slice(0, 10), clf
    )
    with mlflow.start_run():
        scores = model_evaluation(pipeline, classifier, X, y)
        click.echo(scores)
        params = model_tuning(pipeline, classifier, X, y)
        final_model = build_pipeline(
            reduce_dim,
            slice(0, 10),
            init_classifier(classifier)
        )
        final_model.set_params(**params)
        final_model.fit(X, y)
        mlflow.log_metrics(scores)
        mlflow.log_params(params)
        mlflow.log_param("classifier", classifier)
        mlflow.log_param("reduce_dim", reduce_dim)
        mlflow.sklearn.log_model(pipeline, "cla")
    #saving the model
    path_folder = save_model_path.parent
    path_folder.mkdir(exist_ok=True)
    save_model_path.unlink(missing_ok=True)
    dump(final_model, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")