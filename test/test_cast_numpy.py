import pytest
from pytest import mark
from pydantic import validate_call

try:
    import numpy as np
except ImportError:
    pytest.skip("Numpy is not installed, This file was skipped.", allow_module_level=True)

from dataschemas.numpy import NDArray

# dataschemas.numpy.NDArray can cast input to numpy.ndarray
# you can also type hint the dtype you want

@mark.parametrize('arr', [
    [1, 2, 3],
    (1, 2, 3),
    3,
])
@validate_call
def test_cast_float32_numpy(
    arr: NDArray[np.float32],
) -> None:
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32

@mark.parametrize('arr', [
    [1, 2, 3],
    (1, 2, 3),
    3,
])
@validate_call
def test_cast_int64_numpy(
    arr: NDArray[np.int64],
) -> None:
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int64