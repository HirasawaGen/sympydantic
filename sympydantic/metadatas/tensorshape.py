from __future__ import annotations

from typing import Any, cast, Literal, override, TypeVar, Callable
from types import EllipsisType
from functools import singledispatchmethod, reduce

from pydantic_core import core_schema
from sympy import Expr, Symbol  # type: ignore[import-untyped]
import sympy as sp

from .basemetadata import SubscriptableMetadata, _SupportsGetitem
from .protocols import TensorLike
from ..errors.compare import DimensionGeError, DimensionLtError


type SingleTypes = int | str | TypeVar | Expr | EllipsisType
type SliceTypes = int | Expr | None


class _Tensorshape(SubscriptableMetadata):
    # TODO: allow_reshape and allow_broadcast and allow_squeeze
    # PS: I try to implement it, but `validate_call` is not support custom config keys.
    # may be I can use `strict=True/False/None` to represent these options.
    # strict=True: not allow all, and not all list cast to tensorlike.
    # strict=None: allow unsqueeze and squeeze, broadcast
    # strict=False: allow reshape
    # TODO: change all the ErrorType to PydanticUserError & TypeError
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
        strict = config.get('strict', False)
        context = {} if info.context is None else info.context
        if 'sympy_namespace' not in context:
            context['sympy_namespace'] = {}
        sympy_namespace = context['sympy_namespace']
        if 'tensor_shapes' not in context:
            context['tensor_shapes'] = {}
        elif field_name in context['tensor_shapes']:
            raise ValueError(f"tensor shape for {field_name} has already been validated")
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
            if isinstance(dim, str) and dim.startswith('*'):
                name = dim[1:]
                if name not in tensor_shapes:
                    raise ValueError(f"tensor shape for {name} is not defined")
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
            raise ValueError(f"only one ellipsis is allowed in shape, got {ellipsis_count}")
        elif ellipsis_count == 1:
            # tensorshape[X, ..., 2*X]
            # This means the ndim of this tensor should be at least 2
            # and the last dimension should be twice of the first dimension
            # tensorshape[..., 3]
            # This means the last dimension should be 3
            # etc.
            if len(req_shape) >= len(orig_shape):
                raise ValueError(f"shape {orig_shape} is not compatible with required shape {req_shape}")
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
                raise ValueError(f"shape {orig_shape} is not compatible with required shape {req_shape}")
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
                raise ValueError(f"dimension {dim} has length {orig_len}, expected {req_len}")
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
                    # and we found it is 4, it will go into the under branch and raise ValueError
                    if sympy_namespace[req_len.name] != orig_len:
                        raise ValueError(f"Symbol {req_len} is already defined as {sympy_namespace[req_len.name]}, you provide incompatible value: {orig_len}")
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
        raise NotImplementedError(f"unsupported type {type(req_len)} for dimension {dim}")
    
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
            raise ValueError(f"Expression {req_len} has undifined symbols: {diff_symbols}.")
        solved_value = req_len.subs(sympy_namespace)
        if solved_value.is_integer and int(solved_value) == orig_len:
            return
        raise ValueError(f"Expression {req_len} is solved to {solved_value}, you provide incompatible value: {orig_len}")
    
    @_dim_validate.register
    def _(
        self,
        req_len: slice,
        orig_len: int,
        dim: int,
        info: core_schema.ValidationInfo,
        sympy_namespace: dict[str, int]
    ) -> None:
        # TODO: use greater than and less than error message, which is alread provided by pydantic
        begin: SliceTypes = req_len.start
        end: SliceTypes = req_len.stop
        if isinstance(begin, Expr):
            free_symbols = {symbol.name for symbol in begin.free_symbols}
            diff_symbols = free_symbols - sympy_namespace.keys()
            if len(diff_symbols):
                raise ValueError(f"slice start {begin} has free symbols {diff_symbols}, but they are not defined in the shape")
            begin = begin.subs(sympy_namespace)
            if not begin.is_integer:
                raise ValueError(f"slice start {begin} is not an integer")
            begin = int(begin)
        if isinstance(end, Expr):
            free_symbols = {symbol.name for symbol in end.free_symbols}
            diff_symbols = free_symbols - sympy_namespace.keys()
            if len(diff_symbols):
                raise ValueError(f"slice end {end} has free symbols {diff_symbols}, but they are not defined in the shape")
            end = end.subs(sympy_namespace)
            if not end.is_integer:
                raise ValueError(f"slice end {end} is not an integer")
            end = int(end)
        if begin is not None and orig_len < begin:
            raise DimensionGeError(dim, begin, orig_len)
        if end is not None and end < orig_len:
            raise DimensionLtError(dim, end, orig_len)


tensorshape = cast(
    _SupportsGetitem[tuple[SingleTypes | slice, ...], _Tensorshape],
    _Tensorshape.subscriptable()
)

