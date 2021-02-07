import data_evaluation as eval
import numpy as np
import pytest


def test_prediction_model_coef_none_intercept_exists():
    data = np.array([[1, 3, 1], [1, 3, 2], [1, -4, 3], [1, 5, 4]])

    with pytest.raises(ValueError):
        eval.LinearPredictionModel(coef=None, intercept=data[:, 0])


def test_prediction_model_coef_and_intercept_none():
    test_model = eval.LinearPredictionModel(coef=None, intercept=None)

    assert test_model.intercept_ is None and test_model.coef_ is None


def test_prediction_model_coef_exists_intercept_none():
    data = np.array([[1, 3, 1], [1, 3, 2], [1, -4, 3], [1, 5, 4]])
    test_coef = data[:, 1:3]
    test_model = eval.LinearPredictionModel(coef=test_coef, intercept=None)

    assert test_model.intercept_.shape[0] == test_coef.shape[0]


def test_prediction_model_coef_and_intercept_exists():
    data = np.array([[1, 3, 1], [1, 3, 2], [1, -4, 3], [1, 5, 4]])
    test_coef = data[:, 1:3]
    test_intercept = data[:, 0]
    test_model = eval.LinearPredictionModel(coef=data[:, 1:3], intercept=data[:, 0])

    assert test_model.coef_.shape == test_coef.shape and test_model.intercept_.shape == test_intercept.shape


def test_prediction_model_fit_method():
    data = np.array([[1, 3, 1], [1, 3, 2], [1, -4, 3], [1, 5, 4]])
    test_model = eval.LinearPredictionModel(coef=data[:, 1:3], intercept=data[:, 0])

    with pytest.raises(NotImplementedError):
        test_model.fit([1, 2, 3], 1)
