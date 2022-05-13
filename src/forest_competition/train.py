import click
import mlflow
import mlflow.sklearn
import pandas as pd

from pathlib import Path
from joblib import dump

from .data import get_data
from .pipeline import create_pipeline
from .evaluation import evaluate
from .model import generate_model


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True,
                    dir_okay=False,
                    path_type=Path),
    show_default=True,
    help="Path to the dataset file."
)
@click.option(
    "-s",
    "--save-model-path",
    default="model/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
    help="Path where the fitted model will be saved."
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
    help="Parameter to control the random number generator used by algorithms."
)
@click.option(
    "--test-split-ratio",
    default=0.3,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
    help="Fraction of splitting the dataset for training and testing a model."
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
    help="Use StandardScaler to standertize numeric parameters."
)
@click.option(
    "--use-pca",
    default=False,
    type=bool,
    show_default=True,
    help="Use PCA to reduce dimensionality."
)
@click.option(
    "--n-neighbors",
    default=5,
    type=int,
    show_default=True,
    help="Number of neighbors used by KNN algorithm."
)
@click.option(
    "--knn-weights",
    default='uniform',
    type=click.Choice(["uniform", "distance"]),
    show_default=True,
)
@click.option(
    "--lr-c",
    default=1.0,
    type=float,
    show_default=True,
    help="Inverse of regularization strength used by LogisticRegression algorithm."
)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True,
    help="The number of trees in the random forest."
)
@click.option(
    "--criterion",
    default='gini',
    type=click.Choice(["gini", "entropy", "log_loss"]),
    show_default=True,
    help="Random forest function to measure the quality of a split."
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
    show_default=True,
    help="The maximum depth of the random forest tree."
)
@click.option(
    "--classifier",
    default="knn",
    type=click.Choice(["knn", "lr", "rf"]),
    show_default=True,
    help="Algorithm to be used for modeling."
)

def train(
        dataset_path: Path,
        save_model_path: Path,
        random_state: int,
        test_split_ratio : float,
        use_scaler : bool,
        use_pca : bool,
        n_neighbors : int,
        knn_weights : str,
        lr_c : float,
        n_estimators : str,
        criterion : str,
        max_depth : int,
        classifier : str
) -> None:
    """Script to train and save model."""
    X_train, X_test, y_train, y_test = get_data(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    #generate model object
    clf = generate_model(
        classifier, random_state,
        n_neighbors, knn_weights,
        lr_c, n_estimators,
        criterion, max_depth
    )
    #pipeline
    pipeline = create_pipeline(
        use_scaler, use_pca,
        X_train.iloc[:, :10].columns,
        classifier
    )
    pipeline.fit(X_train, y_train)
    scores = evaluate(pipeline, X_test, y_test)
    # logging
    with mlflow.start_run():
        mlflow.log_metrics(scores)
        mlflow.log_param("classifier", classifier)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("use_pca", use_pca)
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("knn_weights", knn_weights)
        mlflow.sklearn.log_model(pipeline, "cla")
    #saving model
    path_folder = save_model_path.parent
    path_folder.mkdir(exist_ok=True)
    save_model_path.unlink(missing_ok=True)
    dump(pipeline, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")