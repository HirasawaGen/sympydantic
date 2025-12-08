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
    (2, 2, 5),  # value_conflict
    # The expression '2*x' is solved as 4.0, which is conflict with the provided value 5.
    (5, 6, 10)  # symbol_redefine
    # The symbol 'x' is already set to 5. you provide a conflict value 6.
])
@invalid_call
@validate_call
def test_symbol_wrong(
    a: Annotated[int, x],
    b: Annotated[int, x],
    c: Annotated[int, x * 2],
) -> NoReturn:
    raise AssertionError('This scope should not be reached')

@mark.parametrize('a', [
    5
])
@invalid_call('symbol_undefined')
@validate_call
def test_symbol_symbol_undefined(
    a: Annotated[int, x+y],
) -> NoReturn:
    # The symbols {x, y} in 'x + y' are not appeared in the above validations.
    raise AssertionError('This scope should not be reached')


@mark.parametrize('a', [
    5
])
@invalid_call
@validate_call
def test_symbol_undefined_single(
    a: Annotated[int, x+1],
) -> NoReturn:
    # The symbol 'x' in 'x + 1' is not appeared in the above validations. When a symbol first appeared, They should be a single symbol format, like 'x'. But 'x + 1' you provide is a complex expression.
    raise AssertionError('This scope should not be reached')