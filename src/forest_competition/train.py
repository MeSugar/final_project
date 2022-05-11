from pathlib import Path
import typing

import click
import pandas as pd

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
    "--max_iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--max_iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-C",
    default=1.0,
    type=float,
    show_default=True,
)
def train(
        dataset_path: Path,
        save_model_path: Path,
        random_state: int,
        test_split_ratio : float
) -> None:
    """Script to train and save model."""
    X_train, X_test, y_train, y_test = get_data(
        dataset_path,
        random_state,
        test_split_ratio,
    )

