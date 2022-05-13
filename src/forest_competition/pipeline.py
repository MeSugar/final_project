import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA


def create_pipeline(
    use_scaler : bool,
    use_pca : bool,
    columns_to_scale : pd.DataFrame,
    classifier
) -> Pipeline:
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler(), columns_to_scale))
    if use_pca:
        steps.append(("pca", PCA(n_components=5), columns_to_scale))
    preprocessor = ColumnTransformer(
    steps,
    remainder="passthrough"
    )
    pipeline = make_pipeline(
        preprocessor,
        classifier)
    return pipeline
