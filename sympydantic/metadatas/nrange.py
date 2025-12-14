from __future__ import annotations

from typing import override, cast, Any, Final

from pydantic_core import core_schema
from sympy import Expr  # type: ignore[import-untyped]

from .basemetadata import SubscriptableMetadata, _SupportsGetitem
from .protocols import _SupportsCompareNumber, _SupportsCompare


# 平泽原你看你代码写得像坨粑粑……  --平泽原


class _NRange(SubscriptableMetadata):
    def __init__(self, range_: slice):
        type RangeType = int | Expr | None
        self._start = cast(RangeType, range_.start)
        self._stop = cast(RangeType, range_.stop)
        self._step = cast(RangeType, range_.step)

    @override
    def _validate(
        self,
        value: _SupportsCompareNumber | Any,
        info: core_schema.ValidationInfo,
    ) -> _SupportsCompareNumber:
        if not isinstance(value, _SupportsCompare):
            raise TypeError(
                f'value must be a number, not {type(value).__name__}'
            )
        value = cast(_SupportsCompareNumber, value)
        config = {} if info.config is None else info.config
        strict = config.get('strict', False)  # noqa: F841
        context = info.context
        if context is None:
            return value
        if 'sympy_namespace' not in context:
            context['sympy_namespace'] = {}
        sympy_namespace = context['sympy_namespace']
        start: None | float
        stop: None | float
        if isinstance(self._start, Expr):
            free_symbols = {symbol.name for symbol in self._start.free_symbols}
            diff_symbols = free_symbols - sympy_namespace.keys()
            if len(diff_symbols):
                raise ValueError(f'undefined symbols: {diff_symbols}')
            start_expr = self._start.subs(sympy_namespace)
            if not start_expr.is_number:
                raise ValueError(f'start must be a number, not {start_expr}')
            start = float(start_expr)
        else:
            start = None if self._start is None else float(self._start)
        if isinstance(self._stop, Expr):
            free_symbols = {symbol.name for symbol in self._stop.free_symbols}
            diff_symbols = free_symbols - sympy_namespace.keys()
            if len(diff_symbols):
                raise ValueError(f'undefined symbols: {diff_symbols}')
            stop_expr = self._stop.subs(sympy_namespace)
            if not stop_expr.is_number:
                raise ValueError(f'stop must be a number, not {stop_expr}')
            stop = float(stop_expr)
        else:
            stop = None if self._stop is None else float(self._stop)
        # TODO: support step

        if start is not None and value < start:  # type: ignore[operator]
            raise ValueError(f'value must be greater than {start}')
        if stop is not None and value >= stop:  # type: ignore[operator]
            raise ValueError(f'value must be less than or equal to {stop}')
        return value


nrange: Final[_NRange] = cast(
    _SupportsGetitem[slice, _NRange],
    _NRange.subscriptable()
)
'''
nrange is NOT a pydantic metadata.

but the subscript value of nrange is a metadata object.

you can use `nrange[3:9]` or `nrange[X:19]`
to validate did a number in the interval.

Example:
>>> from typing import Annotated
>>> from sympy.abc import X
>>> from pydantic import validate_call
>>> from sympydantic import nrange

>>> @validate_call
>>> def foo(num: Annotated[int, nrange[3:9]]):
...     print(num)


>>> foo(5)
5
>>> foo(10)
Traceback (most recent call last):
    ...traceback...
pydantic_core._pydantic_core.ValidationError: 1 validation error for f00
0
  Value error, value must be less than or equal to 9.0

>>> # you can also use sympy expression in nrange
>>> @validate_call
>>> def foo(x: Annotated[int, X], num: Annotated[int, nrange[3:X]]):
...     # pydantic will make sure num is in the interval of [3, x)
...     print(num)

>>> foo(10, 5)
5
>>> foo(10, 15)
Traceback (most recent call last):
    ...traceback...
pydantic_core._pydantic_core.ValidationError: 1 validation error for f00
0
  Value error, value must be less than or equal to 10.0

'''
