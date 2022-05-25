import pytest
import numpy as np
from click.testing import CliRunner
from typing import Any
from forest_competition.train import train
from forest_competition.data import generate_features
from test_helper import generate_data
from joblib import load


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


path_test_args = [("invalid_file", 2), (True, 1), (123, 1)]


@pytest.mark.parametrize("test_input, expected", path_test_args)
def test_train_fails_on_invalid_dataset_path(
    runner: CliRunner, test_input: Any, expected: int
) -> None:
    """It fails when --dataset-path option gets wrong path"""
    result = runner.invoke(train, ["--dataset-path", test_input])
    assert result.exit_code == expected


@pytest.mark.parametrize("test_input, expected", path_test_args)
def test_train_fails_on_invalid_save_model_path(
    runner: CliRunner, test_input: Any, expected: int
) -> None:
    """It fails when --save-model-path option gets wrong path"""
    result = runner.invoke(train, ["--save-model-path", test_input])
    assert result.exit_code == expected


reduce_dim_test_args = [("invalid", 2), (True, 2), (1, 2)]


@pytest.mark.parametrize("test_input, expected", reduce_dim_test_args)
def test_train_fails_on_invalid_reduce_dim_value(
    runner: CliRunner, test_input: Any, expected: int
) -> None:
    """It fails when --reduce-dim option gets wrong value"""
    result = runner.invoke(train, ["--reduce-dim", test_input])
    assert result.exit_code == expected
    assert "Invalid value for '--reduce-dim'" in result.output


invert_dummy_test_args = [
    ("invalid", [2, "Invalid value for '--invert-dummy'"]),
    (1.1, [1, ""]),
    (123, [1, ""]),
]


@pytest.mark.parametrize("test_input, expected", invert_dummy_test_args)
def test_train_fails_on_invalid_invert_dummy_value(
    runner: CliRunner, test_input: Any, expected: Any
) -> None:
    """It fails when --invert-dummy option gets wrong value"""
    result = runner.invoke(train, ["--invert-dummy", test_input])
    assert expected[0] == result.exit_code
    assert expected[1] in result.output


classifier_test_args = [("invalid", 2), ("svm", 2), (True, 2), (1, 2)]


@pytest.mark.parametrize("test_input, expected", classifier_test_args)
def test_train_fails_on_invalid_classifier_value(
    runner: CliRunner, test_input: Any, expected: int
) -> None:
    """It fails when --classifier option gets wrong value"""
    result = runner.invoke(train, ["--classifier", test_input])
    assert result.exit_code == expected
    assert "Invalid value for '--classifier'" in result.output


def test_model(runner: CliRunner) -> None:
    with runner.isolated_filesystem():
        train_data = generate_data(200)
        train_data.to_csv("train.csv", index=False)

        result = runner.invoke(
            train, ["--dataset-path", "train.csv", "--save-model-path", "model.joblib"]
        )
        model = load("model.joblib")
        test_data = generate_data(100).drop(["Id", "Cover_Type"], axis=1)
        test_data = generate_features(test_data)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
        assert np.all(model.predict_proba(test_data) >= 0)
        assert np.all(model.predict_proba(test_data) <= 1)
        assert np.all(model.predict(test_data) >= 1)
        assert np.all(model.predict(test_data) <= 7)
        assert result.exit_code == 0
