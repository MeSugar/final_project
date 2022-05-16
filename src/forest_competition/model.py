import pandas as pd
import numpy as np

from typing import Any
from typing import Dict

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

def init_classifier(classifier : str, random_state : int) -> Any:
    if classifier == 'knn':
        return KNeighborsClassifier(n_jobs=-1)
    elif classifier == 'logreg':
        return LogisticRegression(random_state=random_state, n_jobs=-1)
    elif classifier == 'rfc':
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)

def build_param_grid(classifier : str) -> Dict:
    param_grid = {}
    if classifier == 'knn':
        param_grid['kneighborsclassifier__n_neighbors'] = list(range(1, 16))
        param_grid['kneighborsclassifier__weights'] = ['uniform', 'distance']
        param_grid['kneighborsclassifier__p'] = [1, 2]
    elif classifier == 'logreg':
        param_grid['C'] = [100, 10, 1.0, 0.1, 0.01]
        param_grid['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
    elif classifier == 'rfc':
        param_grid['n_estimators'] = np.linspace(200, 2000, 10)
        arr = np.linspace(10, 110, 11)
        param_grid['max_depth'] = np.append(arr, None)
        param_grid['min_samples_split'] = [2, 5, 10]
        param_grid['min_samples_leaf'] = [1, 2, 4]
    return param_grid

def model_evaluation(
    pipeline : Any,
    clf_name : str,
    X : pd.DataFrame,
    y : pd.Series
):
    cv_inner = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    cv_outer = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )
    param_grid = build_param_grid(clf_name)
    scoring = {
        'accuracy_score' : make_scorer(accuracy_score),
        'f1_score' : make_scorer(f1_score, average='weighted'),
        'roc_auc_score' : make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
    }
    search = RandomizedSearchCV(
        estimator=pipeline, 
        param_distributions=param_grid,
        cv=cv_inner,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    scores = cross_validate(search, X, y, scoring=scoring, cv=cv_outer, n_jobs=-1)
    means = {
        'accuracy' : np.mean(scores["test_accuracy_score"]),
        'f1' : np.mean(scores["test_f1_score"]),
        'roc_auc' : np.mean(scores["test_roc_auc_score"])
    }
    return means

def model_tuning(
    pipeline : Any,
    clf_name : str,
    X : pd.DataFrame,
    y : pd.Series
):
    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )
    param_grid = build_param_grid(clf_name)
    search = RandomizedSearchCV(
        estimator=pipeline, 
        param_distributions=param_grid,
        cv=cv,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    search.fit(X, y)
    return search.best_params_

