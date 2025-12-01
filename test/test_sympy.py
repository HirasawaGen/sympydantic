from typing import Annotated, NoReturn

from pytest import mark
from pydantic import validate_call
from sympy import Symbol, Expr  # type: ignore[import-untyped]
from sympy.abc import x, y, z  # type: ignore[import-untyped]

from . import invalid_call


def test_patch() -> None:
    assert hasattr(Expr, '__get_pydantic_core_schema__')
    assert hasattr(Symbol, '__get_pydantic_core_schema__')


@mark.parametrize('a,b,c', [
    (2, 2, 4),
    (5, 5, 10)
])
@validate_call
def test_symbol_right(
    a: Annotated[int, x],
    b: Annotated[int, x],
    c: Annotated[int, 2*x],
) -> None:
    assert a == b
    assert 2*a == c
    
@mark.parametrize('a,b,c', [
    (2, 2, 5),
    (5, 6, 10)
])
@invalid_call
@validate_call
def test_symbol_wrong(
    a: Annotated[int, x],
    b: Annotated[int, x],
    c: Annotated[int, x * 2],
) -> NoReturn:
    raise AssertionError('This scope should not be reached')