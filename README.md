This is the solution for the [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/) competition formated as a **Python package**. It uses [Forest train](https://www.kaggle.com/competitions/forest-cover-type-prediction/data?select=train.csv) dataset.
![image](https://user-images.githubusercontent.com/75207011/169409249-187012e0-7370-46a4-9427-267b5190dd85.png)


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




