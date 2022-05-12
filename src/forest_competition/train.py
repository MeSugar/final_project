from pathlib import Path

from joblib import dump
from joblib import load
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import click
import pandas as pd
import os

from .get_data import get_data
from .create_pipeline import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True,
                    dir_okay=False,
                    path_type=Path),
    show_default=True
)
@click.option(
    "-s",
    "--save-model-path",
    default="model/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.3,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--n-neighbors",
    default=5,
    type=int,
    show_default=True,
)
@click.option(
    "--knn-weights",
    default='uniform',
    type=str,
    show_default=True,
)

def train(
        dataset_path: Path,
        save_model_path: Path,
        random_state: int,
        test_split_ratio : float,
        max_iter : int,
        n_neighbors : int,
        knn_weights : str
) -> None:
    """Script to train and save model."""
    X_train, X_test, y_train, y_test = get_data(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    pipeline = create_pipeline(n_neighbors, knn_weights)
    pipeline.fit(X_train, y_train)
    scoring = {
        'accuracy_score' : make_scorer(accuracy_score),
        'f1_score' : make_scorer(f1_score, average='weighted'),
        'roc_auc_score' : make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
    }
    for key in scoring:
        score  = cross_val_score(
        pipeline, X_test, y_test, cv=StratifiedKFold(n_splits=5),
        scoring=scoring[key]).mean()
        click.echo(f"{key}: {score}")
    path_folder = save_model_path.parent
    path_folder.mkdir(exist_ok=True)
    save_model_path.unlink(missing_ok=True)
    dump(pipeline, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")
    