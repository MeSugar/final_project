import pytest
import click
from click.testing import CliRunner
from typing import Any
from forest_competition.train import train


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
    runner: CliRunner, test_input: Any, expected: int
) -> None:
    """It fails when --invert-dummy option gets wrong value"""
    result = runner.invoke(train, ["--invert-dummy", test_input])
    click.echo(result.output)
    assert result.exit_code == expected[0]
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
