
[![Tests](https://github.com/mesugar/forest_competition/workflows/Tests/badge.svg)](https://github.com/mesugar/forest_competition/actions?workflow=Tests)

This is the solution for the [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/) competition formated as a **Python package**. It uses [Forest train](https://www.kaggle.com/competitions/forest-cover-type-prediction/data?select=train.csv) dataset.
![image](https://user-images.githubusercontent.com/75207011/169409249-187012e0-7370-46a4-9427-267b5190dd85.png)

## Goals
The purpose of this project is not only to solve the competition, but also to master modern tools that are useful for writing quality code, as well as developing and deploying models.
Used tools:
- Poetry
- MLflow
- pytest
- flake8
- mypy
- black
- nox
- GitHub Actions
- FastAPI
- Docker

## Usage
This package allows you to train model for predicting the forest cover type (the predominant kind of tree cover) from strictly cartographic variables.
1. Clone this repository to your machine:
```
git clone https://github.com/MeSugar/forest_competition.git
cd forest_competition
```
2. Download [Forest train](https://www.kaggle.com/competitions/forest-cover-type-prediction/data?select=train.csv) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.8 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I used Poetry 1.1.13).
4. Install the project dependencies (run this and following commands in a terminal, from the root of a cloned repository):
```
poetry install --no-dev
```
5. Run train with the following command:
```
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (e.g., the algorithm to be chosen for the task) in the CLI. To get a full list of them, use help:
```
poetry run train --help
```
6. To see the information about conducted experiments (algorithm, metrics, hyperparameters) run [MLflow UI](https://mlflow.org/docs/latest/index.html):
```
poetry run mlflow ui
```
![image](https://user-images.githubusercontent.com/75207011/168317447-aba16bc1-32fb-4081-8b05-cfe2c65c9827.png)

7. You can produce EDA report in .html format using [Pandas-Profiling](https://github.com/ydataai/pandas-profiling):
```
poetry run eda -d <path to csv with data> -s <path to save report>
```
To see the list of configure options in the CLI run with *--help* option

8. To make submission file with predictions run:
```
poetry run predict
```
To see the list of configure options in the CLI run:
```
poetry run predict --help
```

## Development
The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. [pytest](https://docs.pytest.org/en/6.2.x/index.html):
```
poetry run pytest
```
![image](https://user-images.githubusercontent.com/75207011/170266137-a4ca82be-3b3b-46e3-af10-9ef3a1d1d9bd.png)

Lint source code with [flake8](https://flake8.pycqa.org/en/latest/):
```
poetry run flake8 src tests noxfile.py
```

Format your code with [black](https://github.com/psf/black):
```
poetry run black src tests noxfile.py
```
![image](https://user-images.githubusercontent.com/75207011/170269669-55424ec8-22f9-4b03-ba31-ba01bfda6517.png)

Perform type cheking with [mypy](https://mypy.readthedocs.io/en/stable/):
```
poetry run mypy src tests noxfile.py
```
![image](https://user-images.githubusercontent.com/75207011/170271050-25faf91f-be8f-4472-801f-9fef0cf600fa.png)

More conveniently, to run all sessions of testing, formatting and type checking in a single command, install and use [nox](https://nox.thea.codes/en/stable/):
```
pip install --user --upgrade nox
nox [-r]
```
![image](https://user-images.githubusercontent.com/75207011/170167653-b0296f85-e820-477c-aaff-111477a1a399.png)

In case you want to run a specific step:
```
nox -[r]s flake8
nox -[r]s black
nox -[r]s mypy
nox -[r]s tests
```

## References
### Model evaluation and selection
- [Article series on model evaluation, model selection, and algorithm selection](https://sebastianraschka.com/blog/2016/model-evaluation-selection-part1.html)
- [Scikit-learn guide to cross-validation](https://scikit-learn.org/stable/model_selection.html)
- [Nested cross-validation](https://weina.me/nested-cross-validation/)

### Tracking experiments
- [ML Experiment Tracking](https://neptune.ai/blog/ml-experiment-tracking)
- [How We Track Machine Learning Experiments with MLFlow](https://www.datarevenue.com/en-blog/how-we-track-machine-learning-experiments-with-mlflow)

### Project organization
- [I don't like notebooks.- Joel Grus](https://www.youtube.com/watch?v=7jiPeIFXb6U)
- [The Complete Guide to Python Virtual Environments!](https://www.youtube.com/watch?v=KxvKCSwlUv8)
- [The Hitchhiker's Guide to Python](https://docs.python-guide.org/writing/structure/)
- [Article series Hypermodern Python](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/)

### Code style, reproducibility, testing
- [Python testing tutorial](https://realpython.com/python-testing/)
- [Carl Meyer - Type-checked Python in the real world](https://www.youtube.com/watch?v=pMgmKJyWKn8)
- [flake8](https://flake8.pycqa.org/en/latest/)
- [Reproducibility, Replicability, and Data Science](https://www.kdnuggets.com/2019/11/reproducibility-replicability-data-science.html)

### Model deployment
- [FastAPI Deployment Tutorials](https://www.youtube.com/playlist?list=PLZoTAELRMXVPgsojPOHF9i0u2L83-m9P7)
- [Model deployment with FastAPI and Docker](https://towardsdatascience.com/how-to-deploy-a-machine-learning-model-with-fastapi-docker-and-github-actions-13374cbd638a)


