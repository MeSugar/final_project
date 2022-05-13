import click
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from typing import Dict
from typing import Any


def evaluate(
    clf : Any,
    X_test : np.array,
    y_test : pd.Series
) -> Dict[str, float]:
    scoring = {
        'accuracy_score' : make_scorer(accuracy_score),
        'f1_score' : make_scorer(f1_score, average='weighted'),
        'roc_auc_score' : make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
    }
    scores = {}
    for key in scoring:
        scores[key] = cross_val_score(
        clf, X_test, y_test, cv=StratifiedKFold(n_splits=5),
        scoring=scoring[key]).mean()
        click.echo(f"{key}: {scores[key]}")
    return scores