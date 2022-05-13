import click
import pandas as pd
import numpy as np

from joblib import dump
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

from typing import List

def create_pipeline(
    use_scaler : bool,
    use_pca : bool,
    columns_to_transorm : List,
    save_pipeline_path
) -> Pipeline:
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler(), columns_to_transorm))
    if use_pca:
        steps.append(("pca", PCA(n_components=5), columns_to_transorm))
    preprocessor = ColumnTransformer(
    steps,
    remainder="passthrough"
    )
    pipeline = make_pipeline(
        preprocessor)
    return pipeline
