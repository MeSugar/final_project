from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(dataset_path: Path,
                random_state: int,
                test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = pd.read_csv(dataset_path, index_col='Id')
    click.echo(f"Dataset shape: {data.shape}.")
    X = data.drop("Cover_Type", axis=1)
    y = data["Cover_Type"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split_ratio, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test