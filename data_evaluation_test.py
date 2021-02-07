import data_evaluation as eval
import numpy as np
import pytest
from constants import Constants


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


def test_handle_one_line_files_true():
    eval.const = Constants()
    eval.const.set_features(2)

    test_data = np.array([1, 2, 3, 40])
    res_w, res_epoch = eval.handle_one_line_files(1, test_data)

    assert res_w.shape == (2, 3)


def test_handle_one_line_files_false():
    eval.const = Constants()
    eval.const.set_features(2)

    test_data = np.array([[1, 2, 3, 40], [1, 2, 3, 60]])
    res_w, res_epoch = eval.handle_one_line_files(2, test_data)

    assert res_w.shape == (2, 3)


def test_get_rounds_with_epoch():
    test_epoch = np.array([[10], [20], [30], [40], [50], [60], [70], [80]])

    result = eval.get_rounds_with_epoch(test_epoch)

    assert result.shape == (8, 2)


def test_get_output_model_norm():
    test_epoch = np.array([[10], [20]])
    test_w = np.array([[1, 2, 3], [1, 5, 6]])

    result = eval.get_output_model_norm(test_w, test_epoch)

    assert result.shape == (2, 2)


def test_get_predict_value():
    test_y = np.array([[0.63], [0.77], [0.35]])
    test_epoch = np.array([[10]])

    result = eval.get_predict_value(test_y, test_y, test_epoch)

    assert result.shape == (1, 2)


def test_get_column_names():
    result = eval.get_column_names('test', 2)

    assert result == ['test_0', 'test_1', 'time']


def test_get_model_error():
    data1 = [[2, 3, 1], [99, 99, 2], [-1, -4, 3], [99, 99, 4]]
    data2 = [[3, 4, 1], [-2, -5, 3]]

    eval.const = Constants()
    eval.const.set_features(1)  # time

    result = eval.get_model_error(data1, data2)

    assert result is not None