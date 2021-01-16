import pytest
import numpy as np


def test_norm_vector_numpy():
    vector = np.array([3, 4, 1])
    assert np.linalg.norm(vector) == 5.0990195135927845


def test_dot_product():
    a = np.array([1, 2, 3])
    b = np.array([1, 5, 7])

    expected = 1*1+2*5+3*7

    assert np.dot(a, b) == expected


def test_divide_vector_with_number():
    vec = np.array([4, 10, 16])
    num = 2

    expected = np.array([4/2, 10/2, 16/2])

    assert bool((vec/num == expected).all()) is True


def test_add_two_vectors():
    a = np.array([1, 2, 3])
    b = np.array([1, 5, 7])

    expected = np.array([1+1, 2+5, 3+7])

    assert bool((np.add(a, b) == expected).all()) is True


def test_max_value():
    a = 5
    b = 10

    expected = 10
    assert max(a, b) == 10
