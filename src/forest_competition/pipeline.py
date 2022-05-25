import pandas as pd

from typing import Any

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


def OneHotInverter(X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
    copy = X.copy()
    copy[X.columns[0] + "_inverted"] = copy.idxmax(1)
    copy.drop(X, axis=1, inplace=True)
    transformed = OrdinalEncoder().fit_transform(copy)
    return transformed


def build_pipeline(
    reduce_dim: str, invert_dummy: bool, columns_to_transorm: slice, clf: Any
) -> Pipeline:
    steps = []
    steps.append(("scaler", StandardScaler(), columns_to_transorm))
    if invert_dummy:
        transformed = FunctionTransformer(OneHotInverter)
        steps.append(("dummyinverter1", transformed, slice(10, 14)))
        steps.append(("dummyinverter2", transformed, slice(14, 54)))
    if reduce_dim == "pca":
        steps.append(("pca", PCA(n_components=0.99), columns_to_transorm))
    preprocessor = ColumnTransformer(steps, remainder="passthrough")
    pipeline = make_pipeline(preprocessor, clf)
    if reduce_dim == "boruta":
        rfc = RandomForestClassifier(n_estimators=1000, n_jobs=3, random_state=42)
        pipeline.steps.insert(
            1,
            (
                "boruta",
                BorutaPy(rfc, n_estimators="auto", verbose=0, random_state=1),
            ),
        )
    return pipeline
