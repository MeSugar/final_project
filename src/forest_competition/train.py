from pathlib import Path
import typing

import click
import pandas as pd

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",  
    type=click.Path(exists=True,
                    dir_okay=False,
                    path_type=Path),
    help='Path to the training dataset.',
    show_default=True
)
@click.option(
    "-s",
    "--save-model-path",
    default="model/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def train(dataset_path: Path) -> None:
    dataset = pd.read_csv(dataset_path)
    click.echo(f"Dataset shape: {dataset.shape}.")