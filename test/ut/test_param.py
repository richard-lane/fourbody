"""
Unit test for param  - basically just test arg verification

"""
import numpy as np
import pytest

from fourbody import param


def test_differnt_particle_numbers():
    """
    Test we get an exception when our arrays contain different particle numbers

    """
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([[2, 2, 3], [3, 4, 4], [5, 6, 5], [7, 8, 6]])

    with pytest.raises(AssertionError):
        param._verify_args(x, x, x, y)

    with pytest.raises(AssertionError):
        param._verify_args(x, x, y, x)

    with pytest.raises(AssertionError):
        param._verify_args(x, y, x, x)

    with pytest.raises(AssertionError):
        param._verify_args(y, x, x, x)

    # Probably no need to test every other combination as well...


def test_not_4_arrays():
    """
    Test we get an exception when our 4 momm arrays dont have 4 components

    """
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [1, 2]])

    with pytest.raises(AssertionError):
        param._verify_args(x, x, x, y)

    with pytest.raises(AssertionError):
        param._verify_args(x, x, y, x)

    with pytest.raises(AssertionError):
        param._verify_args(x, y, x, x)

    with pytest.raises(AssertionError):
        param._verify_args(y, x, x, x)
