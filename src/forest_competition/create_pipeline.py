from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipeline(
    max_iter: int,
    C: float,
    random_state: int
) -> Pipeline:
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            C=C,
            random_state=random_state))
    return pipeline
