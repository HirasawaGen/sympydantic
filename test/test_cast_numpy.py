import pytest
from pytest import mark
from pydantic import validate_call

try:
    import numpy as np
except ImportError:
    pytest.skip("Numpy is not installed, This file was skipped.", allow_module_level=True)

from sympydantic.dataschemas.numpy import NDArray

# dataschemas.numpy.NDArray can cast input to numpy.ndarray
# you can also type hint the dtype you want

@mark.parametrize('arr', [
    [1, 2, 3],
    (1, 2, 3),
    3,
])
# pydantic 装饰器@validate_call 除了可以对数据进行验证，还会对部分错误类型进行自动转化
# 例如函数签名为func(a: int, b: str)，传参a: str = '123', b: int = 456
# 则会自动将'123'转为123，b = 456
# 本项目使用了同样的思路
# 若函数签名为func(a: NDArray[np.int8]), 传参a: list[float] = [1.0, 2.0, 3.0]
# 则会自动将list[float]转为numpy.ndarray[np.int8]
# 若使用validate_call(strict=True)显式标明使用严格模式，则会报错
@validate_call
def test_cast_int8_numpy(
    # 注意：这里的NDArray并不是numpy.typing.NDArray
    # 而是sympydantic.dataschemas.numpy.NDArray
    # 但是sympydantic.dataschemas.numpy.NDArray继承自numpy.typing.NDArray
    # 所以IDE会自动补全ndarray的field，包括dtype max argmax等
    arr: NDArray[np.int8],
) -> None:
    # 若传参不是numpy数组类型，会自动通过np.array转为numpy数组
    assert isinstance(arr, np.ndarray)
    # 若传参是numpy数组类型，但dtype不匹配，会自动通过astype转为匹配的dtype
    assert arr.dtype == np.int8


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