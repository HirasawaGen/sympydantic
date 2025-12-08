from typing import Annotated, NoReturn

from pytest import mark
from pydantic import validate_call
from sympy.abc import X, Y, Z  # type: ignore[import-untyped]

from sympydantic import TensorLike, tensorshape

from . import invalid_call, create_tensor


@mark.parametrize('arr', [
    create_tensor((3, 3, 3))
])
@invalid_call('dimension_conflict')
@validate_call
def test_dimension_conflict_1(
    arr: Annotated[TensorLike, tensorshape[3, 4]]
) -> NoReturn:
    raise AssertionError('This scope should never be reach.')


@mark.parametrize('arr', [
    create_tensor((3, 4, 6))
])
@invalid_call('dimension_conflict')
@validate_call
def test_dimension_conflict_2(
    arr: Annotated[TensorLike, tensorshape[3, 4, ..., 5, 6]]
) -> NoReturn:
    raise AssertionError('This scope should never be reach.')


@mark.parametrize('arr', [
    create_tensor((3, 4))
])
@invalid_call('shape_validated')
@validate_call
def test_shape_validated(
    arr: Annotated[TensorLike, tensorshape[3, 4], tensorshape[3, 3, 3]]
) -> NoReturn:
    raise AssertionError('This scope should never be reach.')


@mark.parametrize('arr', [
    create_tensor((5, 6, 3))
])
@invalid_call('shape_unvalidated')
@validate_call
def test_shape_unvalidated(
    arr: Annotated[TensorLike, tensorshape['*arr2', 3]]
) -> NoReturn:
    raise AssertionError('This scope should never be reach.')

@mark.parametrize('arr', [
    create_tensor((5, 6, 3))
])
@invalid_call('shape_format')
@validate_call
def test_shape_format_1(
    arr: Annotated[TensorLike, tensorshape[..., ...]]
) -> NoReturn:
    raise AssertionError('This scope should never be reach.')


@mark.parametrize('arr', [
    create_tensor((1, 2, 3))
])
@invalid_call('shape_format')
@validate_call
def test_shape_format_2(
    arr: Annotated[TensorLike, tensorshape[1, 2, {'a': 3, 'b': 4}]]
) -> NoReturn:
    raise AssertionError('This scope should never be reach.')


@mark.parametrize('num, arr', [
    [3, create_tensor((4, 3))],
])
@invalid_call('shape_conflict')
@validate_call
def test_shape_conflict(
    num: Annotated[int, X],
    arr: Annotated[TensorLike, tensorshape[3, X]]
) -> NoReturn:
    raise AssertionError('This scope should never be reach.')


@mark.parametrize('num, arr', [
    [3, create_tensor((3, 4))],
])
@invalid_call('symbol_redefined')
@validate_call
def test_symbol_redefined(
    num: Annotated[int, X],
    arr: Annotated[TensorLike, tensorshape[3, X]]
) -> NoReturn:
    raise AssertionError('This scope should never be reach.')

@mark.parametrize('arr', [
    create_tensor((3, 4))
])
@invalid_call('expr_solve')
@validate_call
def test_expr_solve(
    arr: Annotated[TensorLike, tensorshape[X, X/0]]
) -> NoReturn:
    raise AssertionError('This scope should never be reach.')

@mark.parametrize('arr', [
    create_tensor((3, 4, 5))
])
@invalid_call('expr_conflict')
@validate_call
def test_expr_conflict(
    arr: Annotated[TensorLike, tensorshape[X, Y, X+Y]]
) -> NoReturn:
    raise AssertionError('This scope should never be reach.')

# TODO: test cases for slice