from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipeline(
    n_neighbors : int,
    knn_weights : str
) -> Pipeline:
    pipeline = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=knn_weights
            ))
    return pipeline
