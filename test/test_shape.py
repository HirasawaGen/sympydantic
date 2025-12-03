from typing import Annotated, NoReturn

from pytest import mark
from pydantic import validate_call
from sympy.abc import X, Y, Z, alpha, beta  # type: ignore[import-untyped]

from sympydantic import tensorshape, TensorLike

from . import create_tensor, invalid_call


@mark.parametrize(
    "arr",
    [create_tensor((8, 8))]
)
@validate_call
def test_int_right(
    # arr's shape is required to be (8, 8)
    # right answer provide, test pass
    arr: Annotated[TensorLike, tensorshape[8, 8]]
) -> None:
    assert arr.shape == (8, 8)


@mark.parametrize("arr", [
    create_tensor((7, 7, 10)),
    create_tensor((4, 4, 4, 4, 4, 4, 4)),
])
@invalid_call
@validate_call
def test_int_wrong(
    # arr's shape is required to be (8, 8)
    # wrong answer provide, raise `pydantic.ValidationError`
    arr: Annotated[TensorLike, tensorshape[8, 8]]
) -> NoReturn:
    raise AssertionError("should not reach here")


@mark.parametrize("arr",[
    create_tensor((7, 7, 19)),
    create_tensor((3, 4, 13)),
])
@validate_call
def test_slice_right(
    # arr's shape is required to be:
    # 0-th dimension is in [0, 9)
    # 1-th dimension is any value
    # 2-th dimension is greater than twice of 1-th dimension
    # right answer provide, test pass
    arr: Annotated[TensorLike, tensorshape[:9, X, 2*X:]]
) -> None:
    shape = arr.shape
    assert len(shape) == 3
    assert shape[0] < 9
    assert 2 * shape[1] <= shape[2]


@mark.parametrize("arr", [
    create_tensor((10, 7, 19)),  # 0-th dimension not <9
    create_tensor((7, 7, 9)),  # 2-th dimension not >2*1-th dimension
])
@invalid_call
@validate_call
def test_slice_wrong(
    # arr's shape is required to be ... (same as above
    # wrong answer provide, raise `pydantic.ValidationError`
    arr: Annotated[TensorLike, tensorshape[:9, X, 2*X:]]
) -> NoReturn:
    raise AssertionError("should not reach here")



@mark.parametrize("arr", [
    create_tensor((8, 8)),
])
@validate_call
def test_typevar_right[_X](
    arr: Annotated[TensorLike, tensorshape[_X, _X]]
) -> None:
    assert arr.shape[0] == arr.shape[1]


@mark.parametrize("arr", [
    create_tensor((8, 9)),
])
@invalid_call
@validate_call
def test_typevar_wrong[_X](
    arr: Annotated[TensorLike, tensorshape[_X, _X]]
) -> NoReturn:
    raise AssertionError("should not reach here")



@mark.parametrize("arr1, arr2, arr3", [
    [
        create_tensor((8, 8)), 
        create_tensor((1, 3, 8, 8, 6)),
        create_tensor((6, 8, 8))
    ]
])
# validate_call 是pydantic提供的装饰器，用于参数验证
@validate_call
def test_multiple_shape_right(
    # Annotated是python标准库typing模块中的特殊类型标注
    # TensorLike 是自定义协议，torch.tensor和numpy.ndarray都满足这个协议
    # tensorshape是自定义元数据，用于标注满足TensorLike协议的对象的shape字段
    # ----------------------------------------------------- #
    # ... 表示任意尺寸
    arr1: Annotated[TensorLike, tensorshape[...]],
    # X为sympy.Symbol对象，首次出现时为任意int型
    # '*arr1' 表示arr1的shape
    # 即 arr2.shape[2:-1] == arr1.shape
    # `3:` 表示最后一个维度要大于等于3
    arr2: Annotated[TensorLike, tensorshape[1, X, '*arr1', 3:]],
    # X 在arr2的标注中出现过了，所以arr3.shape必须是X+3
    # 即 arr3.shape[0] == arr2.shape[1] + 3
    arr3: Annotated[TensorLike, tensorshape[X + 3, '*arr1']],
) -> None:
    shape1 = arr1.shape
    shape2 = arr2.shape
    shape3 = arr3.shape
    # 若三个参数不满足以下assert
    # @validate_call会抛出pydantic.ValidationError
    # test_multiple_shape_right函数体不会执行
    assert len(shape1) + 3 == len(shape2)
    assert len(shape1) + 1 == len(shape3)
    assert shape2[2: -1] == shape1
    assert shape2[0] == 1
    assert shape2[-1] >= 3
    assert shape3[1:] == shape1
    symbol_X = shape2[1]
    assert shape3[0] == symbol_X + 3
'''
正例：
    arr1=np.ndarray((2, 3))
    arr2=np.ndarray((1, 3, 2, 3, 7))
    arr3=np.ndarray((6, 2, 3))
反例：
    arr1=np.ndarray((2, 3))
    arr2=np.ndarray((2, 3, 2, 3, 7))  # <--- 维度不匹配 标注第一维为1，实际为2
    arr3=np.ndarray((6, 2, 3))
    
    arr1=np.ndarray((2, 3))
    arr2=np.ndarray((1, 3, 2, 3, 7))
    arr3=np.ndarray((6, 3, 3))  # <--- 维度不匹配 标注最后几维要与arr1一致，实际并不一致
    
    arr1=np.ndarray((2, 3))
    arr2=np.ndarray((1, 3, 2, 3, 2))  # <--- 维度不匹配 标注最后一维要大于等于3，实际为2
    arr3=np.ndarray((6, 2, 3))
'''



    



