# flake8: noqa: F403, F405
from __future__ import annotations

from typing import Any, TypeVar, Final
from typing import cast, override
from types import EllipsisType
from functools import singledispatchmethod

from pydantic_core import core_schema
from sympy import Expr, Symbol  # type: ignore[import-untyped]
import sympy as sp

from .basemetadata import SubscriptableMetadata, _SupportsGetitem
from .protocols import TensorLike
from ..errors.compare import *
from ..errors.tensorlike import *
from ..errors.sympy import *


__all__ = ['tensorshape']


type SingleTypes = int | str | TypeVar | Expr | EllipsisType
type SliceTypes = int | Expr | None

_STRICT_MODE: Final[int] = 1
_LAX_MODE: Final[int] = -1
_DEFAULT_MODE: Final[int] = 0


class _Tensorshape(SubscriptableMetadata):
    def __init__(self, shape: tuple[SingleTypes | slice, ...] | SingleTypes | slice):
        if not isinstance(shape, tuple):
            shape = (shape, )
        self._shape: tuple[SingleTypes | slice, ...] = shape
        
    @override
    def _validate(
        self,
        value: TensorLike | Any,
        info: core_schema.ValidationInfo,
    ) -> TensorLike | Any:
        field_name = info.field_name
        if field_name is None:
            return value
        config = {} if info.config is None else info.config
        strict = config.get('strict') == True
        # TODO: use mode, not strict
        # strict mode: not allow all, and not all list cast to tensorlike.
        # default mode: allow unsqueeze and squeeze, broadcast
        # lax mode: allow reshape
        mode: int
        match config.get('strict'):
            case True:  mode = _STRICT_MODE
            case False: mode = _LAX_MODE
            case _:     mode = _DEFAULT_MODE
        context = {} if info.context is None else info.context
        if 'sympy_namespace' not in context:
            context['sympy_namespace'] = {}
        sympy_namespace = context['sympy_namespace']
        if 'tensor_shapes' not in context:
            context['tensor_shapes'] = {}
        elif field_name in context['tensor_shapes']:
            raise ShapeValidatedError(field_name)
        tensor_shapes = context['tensor_shapes']
        if not isinstance(value, TensorLike):
            if not strict:
                return value
            raise TypeError(f"value must be a tensor-like object, got {type(value)}")
        req_shape = self._shape
        orig_shape = value.shape
        tensor_shapes[field_name] = orig_shape
        
        # TODO: this algorithm is pretty good, but it should be move on NDArray dataschema.
        req_shape_list: list[SingleTypes | slice | tuple[int, ...]] = []
        j = 0
        for dim in req_shape:
            if isinstance(dim, str) and dim != '*' and dim.startswith('*'):
                name = dim[1:]
                if name not in tensor_shapes:
                    raise ShapeUnvalidatedError(name)
                if not len(req_shape_list):
                    req_shape_list = list(req_shape)
                req_shape_list[j:j+1] = tensor_shapes[name]
                j += len(tensor_shapes[name])
            j += 1
                
        if len(req_shape_list):
            req_shape = tuple(req_shape_list)
        del req_shape_list
            
        ellipsis_count = req_shape.count(...)  # count the number of ellipsis
        
        if ellipsis_count > 1:
            raise ShapeFormatError('multiple `...`(ellipsis) was passed.')
        elif ellipsis_count == 1:
            # tensorshape[X, ..., 2*X]
            # This means the ndim of this tensor should be at least 2
            # and the last dimension should be twice of the first dimension
            # tensorshape[..., 3]
            # This means the last dimension should be 3
            # etc.
            if len(req_shape) - 1 > len(orig_shape):
                raise VarDimentionConflictError(req_shape, orig_shape)
            ellipsis_idx = req_shape.index(...)
            # replace the ellipsis with the `*`
            # because `*` means any length
            # and make len(req_shape) == len(orig_shape)
            req_shape = (
                *req_shape[:ellipsis_idx],
                *['*'] * (len(orig_shape) - len(req_shape) + 1),
                *req_shape[ellipsis_idx+1:]
            )
        else:  # ellipsis_count == 0
            # if there isn't ellipsis in the shape,
            # the length of the shape should be the same as the tensor's ndim
            if len(req_shape) != len(orig_shape):
                raise DimensionConflictError(req_shape, orig_shape)
        # ellipsis has been handled, so cast the type.
        req_shape = cast(tuple[int | str | Expr | slice, ...], req_shape)
        # get the sympy namespace from the context, this context is get from ValidationInfo
        
        # iter both the required shape and the original shape
        # check if the length of each dimension is valid
        for dim, (req_len, orig_len) in enumerate(zip(req_shape, orig_shape)):
            if req_len == '*' or req_len == orig_len:
                # `*` is any length, so continue
                # if req_len == orig_len, also continue
                continue
            if isinstance(req_len, int):
                # if req_len == orig_len, it will be dealed on the obove if statement
                # So we can make sure req_len != orig_len here
                # and if req_len != orig_len, and req_len is an integer, it means it is a fixed length
                raise ShapeConflictError(dim, f'{req_len} (int)', orig_len)
            if isinstance(req_len, str):
                # string objects who is not `*`, will treated as sympy.Expr object
                req_len = sp.sympify(req_len)
            # req_len variable is now an sp.Expr object
            # do a cast.
            if isinstance(req_len, TypeVar):
                req_len = sp.sympify(req_len.__name__)
            req_len = cast(Expr | slice, req_len)
            if isinstance(req_len, Symbol):
                # Symbol object is like: `X`, `Y`, etc.
                # `X+1`, `X+Y`, `2*X/Y` will never into this branch.
                if req_len.name not in sympy_namespace:
                    # first time to see this symbol, just assign the length
                    # for example, tensorshape[X, X], but you provide shape is (3, 4)
                    # first time see `X`, will assign X = 3 into the sympy_namespace
                    sympy_namespace[req_len.name] = orig_len
                else:
                    # if this Symbol obj is already seen,
                    # it will save in the sympy_namespace,
                    # so it will into this branch.
                    # like the above example, tensorshape[X, X], provide shape is (3, 4)
                    # X is already seen and set to 3, so it will check if the length is 3
                    # and we found it is 4, it will go into the under branch and raise error
                    if sympy_namespace[req_len.name] != orig_len:
                        raise SymbolRedefinedError(req_len, sympy_namespace[req_len.name], orig_len)
                continue
            # req_len is an sp.Expr object, but not sp.Symbol, because Symbol object is dealed in the previous branch
            # function calling is super slow, so I have to use awful if-else.
            self._dim_validate(req_len, orig_len, dim, info, sympy_namespace)
        return value
    

    
    @singledispatchmethod
    def _dim_validate(
        self,
        req_len: Expr | slice,
        orig_len: int,
        dim: int,
        info: core_schema.ValidationInfo,
        sympy_namespace: dict[str, int]
    ) -> None:
        raise ShapeFormatError(f"unsupported type {type(req_len)} for dimension {dim}")
    
    @_dim_validate.register
    def _(
        self,
        req_len: Expr,
        orig_len: int,
        dim: int,
        info: core_schema.ValidationInfo,
        sympy_namespace: dict[str, int]
    ) -> None:
        # if req_len is an sp.Symbol object, it will be deal in the previous register
        # So in this case, we just need to check if the expression is valid
        free_symbols = {symbol.name for symbol in req_len.free_symbols}
        diff_symbols = free_symbols - sympy_namespace.keys()
        if len(diff_symbols):
            raise SymbolUndefinedError(diff_symbols, req_len)
        solved_value = req_len.subs(sympy_namespace)
        if not solved_value.is_integer:
            raise ExpressionSolveError(req_len, solved_value)
        if not int(solved_value) == orig_len:
            raise ExpressionConflictError(req_len, solved_value, orig_len)
    
    @_dim_validate.register
    def _(
        self,
        req_len: slice,
        orig_len: int,
        dim: int,
        info: core_schema.ValidationInfo,
        sympy_namespace: dict[str, int]
    ) -> None:
        begin: SliceTypes = req_len.start
        end: SliceTypes = req_len.stop
        if isinstance(begin, Expr):
            free_symbols = {symbol.name for symbol in begin.free_symbols}
            diff_symbols = free_symbols - sympy_namespace.keys()
            if len(diff_symbols):
                raise SymbolUndefinedError(diff_symbols, begin)
            begin_solved = begin.subs(sympy_namespace)
            if not begin_solved.is_integer:
                raise ExpressionSolveError(begin, begin_solved)
            begin = int(begin_solved)
        if isinstance(end, Expr):
            free_symbols = {symbol.name for symbol in end.free_symbols}
            diff_symbols = free_symbols - sympy_namespace.keys()
            if len(diff_symbols):
                raise SymbolUndefinedError(diff_symbols, end)
            end_solved = end.subs(sympy_namespace)
            if not end_solved.is_integer:
                raise ExpressionSolveError(end, end_solved)
            end = int(end_solved)
        if begin is not None and orig_len < begin:
            raise DimensionGeError(dim, begin, orig_len)
        if end is not None and end < orig_len:
            raise DimensionLtError(dim, end, orig_len)


tensorshape: Final[_Tensorshape] = cast(
    _SupportsGetitem[tuple[SingleTypes | slice, ...], _Tensorshape],
    _Tensorshape.subscriptable()
)
'''
tensorshape is NOT a pydantic metadata

but the subscript value of tensorshape is a metadata object.

you can use `tensorshape[X, X:Y]` or `tensorshape[3, 4:9, 10:X]` to validate the shape of tensor-like object.

Example:

>>> from typing import Annotated
>>> import numpy as np
>>> from sympy.abc import X, Y, Z
>>> from pydantic import validate_call
>>> from sympydantic import tensorshape, TensorLike

>>> @validate_call  # you can use sympy in the tensorshape
>>> def foo(a: Annotated[TensorLike, tensorshape[X, X+3]]):
...     print(a.shape)

>>> foo(np.zeros((3, 6)))  # right shape which same with `(X, X+3)`
(3, 6)
>>> foo(np.zeros((3, 7)))  # wrong shape which is `(X, X+4)`
Traceback (most recent call last):
    ...
pydantic_core._pydantic_core.ValidationError: 1 validation error for foo
0
  The expression 'X + 3' is solved as 6, which is conflict with the provided value 7.
  ...

>>> # `Annotated[TensorLike, tensorshape[X, X+3]]` is too long, you can use typealias:
>>> type SquareMatrix = Annotated[TensorLike, tensorshape[X, X]]
>>> type RowVector = Annotated[TensorLike, tensorshape['*', 1]]
>>> type ColVector = Annotated[TensorLike, tensorshape[1, '*']]
>>> type UnsqeezeSelf = Annotated[TensorLike, tensorshape['*self', 1]]

>>> # If you need shape validate is only an interval, you can use slice object.
>>> @validate_call
>>> def foo(a: Annotated[TensorLike, tensorshape[3, X, 9:X*2]]):
...     print(a.shape)
...     # pydantic will guard shape of a as `(3, X, 9:X*2+3)`
...     # equivalent to the under assertion
...     assert a.ndim == 3
...     assert a.shape[0] == 3
...     num_X = a.shape[1]
...     assert 9 <= num_X < 2*num_X + 3

>>> # If you don't want to validate ndim or a specific dimension, you can use `...` or `*`.
>>> @validate_call
>>> def foo(
        a: Annotated[TensorLike, tensorshape['*', '*', '*', '*', '*']],
        b: Annotated[TensorLike, tensorshape[3, ..., 6]],
... ):
...     print(a.shape)
...     print(b.shape)
...     # equivalent to the under assertion
...     assert a.ndim == 5
...     assert b.ndim >= 2  # not validate specific ndim.
...     assert b.shape[0] == 3
...     assert b.shape[-1] == 6

>>> # If you need shape of these arguments have relation with each other, use str as '*xxx' format.
>>> @validate_call
>>> def foo(
        a: Annotated[TensorLike, tensorshape[...]],
        b: Annotated[TensorLike, tensorshape[3, '*a', 4]],
... ):
...     print(a.shape)
...     print(b.shape)
...     # equivalent to the under assertion
...     assert b.ndim - a.ndim == 2
...     assert b.shape[1:-1] == a.shape
...     assert b.shape[0] == 3
...     assert b.shape[-1] == 4
'''
