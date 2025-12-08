from typing import Container, override, final

from sympy import Expr, Symbol
from pydantic_core import PydanticCustomError


__all__ = [
    'SymbolRedefinedError',
    'SymbolUndefinedError',
    'ExpressionSolveError',
    'ExpressionConflictError'
]


@final
class SymbolRedefinedError(PydanticCustomError):
    _error_type: str = 'symbol_redefined'
    _message_template: str = "The symbol '{symbol}' is already set to {solved}. you provide a conflict value {value}."
    
    @override
    def __new__(cls, symbol: Symbol | str, solved: float | int, value: float | int):
        if isinstance(symbol, Symbol):
            symbol = symbol.name
        return super().__new__(
            cls,
            cls._error_type,
            cls._message_template,
            {
                'symbol': symbol,
                'solved': solved,
                'value': value
            }
        )


@final
class SymbolUndefinedError(PydanticCustomError):
    _error_type: str = 'symbol_undefined'
    _message_template = "The symbols {{symbols}} in '{expr}' are not appeared in the above validations."
    _message_template_single = "The symbol '{symbols}' in '{expr}' is not appeared in the above validations."\
                                "When a symbol first appeared, They should be a single symbol format, like '{symbols}'. But '{expr}' you provide is a complex expression."
    
    @override
    def __new__(cls, symbols: Container[Symbol | str] | Symbol | str, expr: Expr):
        is_single = False
        if not isinstance(symbols, Container):
            is_single = True
            symbols = [symbols]
        elif len(symbols) == 1:
            is_single = True
        symbols = ', '.join(str(s) for s in symbols)
        return super().__new__(
            cls,
            cls._error_type,
            cls._message_template_single if is_single else cls._message_template,
            {
                'symbols': symbols,
                'expr': str(expr)
            }
        )


@final
class ExpressionSolveError(PydanticCustomError):
    _error_type: str ='expr_solve'
    _message_template = "The expression '{expr}' is solved as {solved}, which is not a number."
    
    @override
    def __new__(cls, expr: Expr, solved: Expr):
        return super().__new__(
            cls,
            cls._error_type,
            cls._message_template,
            {
                'expr': expr,
                'solved': solved
            }
        )

@final
class ExpressionConflictError(PydanticCustomError):
    _error_type: str = 'expr_conflict'
    _message_template = "The expression '{expr}' is solved as {solved}, which is conflict with the provided value {value}."
    
    @override
    def __new__(cls, expr: Expr, solved: float | int | Expr, value: float | int):
        return super().__new__(
            cls,
            cls._error_type,
            cls._message_template,
            {
                'expr': expr,
                'solved': solved,
                'value': value
            }
        )

