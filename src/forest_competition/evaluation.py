import click
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from typing import Tuple


def evaluate(
    pipeline : Pipeline,
    X_test : pd.DataFrame,
    y_test : pd.Series
) -> Tuple[float]:
    scoring = {
        'accuracy_score' : make_scorer(accuracy_score),
        'f1_score' : make_scorer(f1_score, average='weighted'),
        'roc_auc_score' : make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
    }
    scores = Tuple()
    for i, key in enumerate(scoring):
        scores.append(cross_val_score(
        pipeline, X_test, y_test, cv=StratifiedKFold(n_splits=5),
        scoring=scoring[key]).mean())
        click.echo(f"{key}: {scores[i]}")