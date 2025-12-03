import timeit

import pytest

try:
    import torch
    import numpy as np
except ImportError:
    pytest.skip("Numpy is not installed, This file was skipped.", allow_module_level=True)
    
from annotated_types import Annotated
from pydantic import validate_call
from sympy.abc import X, Y, Z  # type: ignore[import-untyped]

from sympydantic import tensorshape
from sympydantic.metadatas.device import CUDA
from sympydantic.dataschemas.numpy import NDArray
from sympydantic.dataschemas.torch import Tensor


number = 10_0000


@validate_call
def foo(
    arr1: 
        NDArray[np.bool_],
    arr2:
        Annotated[
            Tensor,
            CUDA,
            tensorshape[X, ..., X*2],
        ],
    num: int
) -> int:
    # print(arr2.requires_grad)
    # print(arr1)
    # print(arr1.dtype)
    # print(arr2.device)
    return '12'  # type: ignore[return-value]


def test_timeit():
    print('\nRunning timeit tests with right args...')
    right_args = (
        np.array([[1, 1, 1]] * 3),
        torch.ones((4, 5, 6, 7, 8), device='cuda'),
        1
    )
    times_wrap = timeit.timeit(lambda: foo.__wrapped__(*right_args), number=number)  # type: ignore[arg-type, attr-defined]
    print(f'time without @validate_call: {times_wrap}')
    times_deco = timeit.timeit(lambda: foo(*right_args), number=number)
    print(f'time with @validate_call: {times_deco}')
    
    print('\nRunning timeit tests with wrong args...')
    wrong_args = (
        [[1, 1, 1]] * 3,
        np.random.random((4, 5, 6, 7, 8)),
        '1'
    )
    times_wrap_wrong = timeit.timeit(lambda: foo.__wrapped__(*wrong_args), number=number)  # type: ignore[arg-type, attr-defined]
    print(f'time without @validate_call (wrong args): {times_wrap_wrong}')
    times_deco_wrong = timeit.timeit(lambda: foo(*wrong_args), number=number)  # type: ignore[arg-type]
    print(f'time with @validate_call (wrong args): {times_deco_wrong}')  # almost 1 min...
    

if __name__ == '__main__':
    test_timeit()