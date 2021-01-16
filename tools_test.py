import tools as tl
import numpy as np
import pytest


def test_array_to_vector():
    array = np.random.rand(3, 1)
    res = tl.array_to_vector(array)

    assert res.shape == (3,)


def test_array_to_vector_wrong_type_exception_TypeError():
    with pytest.raises(TypeError):
        tl.array_to_vector([1, 2, 4])


def test_array_to_vector_wrong_dimension_exception_TypeError():
    array = np.random.rand(3, 2)
    with pytest.raises(TypeError):
        tl.array_to_vector(array)


