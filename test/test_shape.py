from typing import Annotated, NoReturn

from pytest import mark
from pydantic import validate_call
from sympy.abc import X, Y, Z, alpha, beta  # type: ignore[import-untyped]

from metadatas import tensorshape
from metadatas.protocols import TensorLike

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




    



