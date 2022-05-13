from typing import Any

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def generate_model(
    classifier : str,
    random_state : int,
    n_neighbors : int,
    knn_weights : str,
    lr_c : float,
    n_estimators : int,
    criterion : str,
    max_depth : int
) -> Any:
    if classifier == 'knn':
        return KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=knn_weights
        )
    elif classifier == 'lr':
        return LogisticRegression(
            C=lr_c,
            random_state=random_state
        )
    else:
        return RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth
        )


