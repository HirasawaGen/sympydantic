from typing import Annotated, NoReturn

from pytest import mark
from pydantic import validate_call
from sympy.abc import X, Y, Z  # type: ignore[import-untyped]

from sympydantic import nrange, tensorshape
from sympydantic import TensorLike
from . import invalid_call, create_tensor


@mark.parametrize("begin, end, num", [
    (3, 5, 4),
    (1, 10, 5),
    (1, 10, 1),
])
@validate_call
def test_nrange_right0(
    begin: Annotated[int, X],
    end: Annotated[int, Y],
    num: Annotated[int, nrange[X:Y]]
) -> None:
    assert begin <= num < end


@mark.parametrize("a, b, c", [
    (5, 6, 4)
])
@validate_call
def test_nrange_right1(
    a: Annotated[int, X],
    b: Annotated[int, nrange[X:]],
    c: Annotated[int, nrange[:X]],
) -> None:
    assert a <= b
    assert c < a
    
    
@mark.parametrize("begin, end, num", [
    (3, 5, 6),
    (3, 5, 2),
])
@invalid_call
@validate_call
def test_nrange_wrong0(
    begin: Annotated[int, X],
    end: Annotated[int, Y],
    num: Annotated[int, nrange[X:Y]]
) -> NoReturn:
    raise AssertionError("Should not reach here")


@mark.parametrize("a, b, c", [
    (5, 4, 6)
])
@invalid_call
@validate_call
def test_nrange_wrong1(
    a: Annotated[int, X],
    b: Annotated[int, nrange[X:]],
    c: Annotated[int, nrange[:X]],
) -> NoReturn:
    raise AssertionError("Should not reach here")


type SquareMatrix_X = Annotated[TensorLike, tensorshape[X, X]]
type SquareMatrix_2X = Annotated[TensorLike, tensorshape[2*X, 2*X]]


@mark.parametrize("size, mat0, mat1", [
    [3, create_tensor((3, 3)), create_tensor((6, 6))],
])
@validate_call
def test_nrange_right2(
    size: Annotated[int, X],
    mat0: Annotated[TensorLike, tensorshape[X, X]],
    mat1: Annotated[TensorLike, tensorshape[2*X, 2*X]],
) -> None:
    assert mat0.shape == (size, size)
    assert mat1.shape == (2*size, 2*size)

@mark.parametrize("size, mat0, mat1", [
    [3, create_tensor((3, 3)), create_tensor((3, 3))],
    [3, create_tensor((3, 4)), create_tensor((6, 6))],
    [3, create_tensor((3, 4)), create_tensor((4, 5))],
])
@invalid_call
@validate_call
def test_nrange_wrong2(
    size: Annotated[int, X],
    mat0: Annotated[TensorLike, tensorshape[X, X]],
    mat1: Annotated[TensorLike, tensorshape[2*X, 2*X]],
) -> NoReturn:
    raise AssertionError("Should not reach here")