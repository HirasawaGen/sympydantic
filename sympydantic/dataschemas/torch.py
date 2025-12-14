from functools import partial
from typing import Any, NoReturn

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

# 我希望最终发行版并不刚需pytorch作为依赖，所以这里这样写了
try:
    import torch
except ImportError:
    raise ImportError('PyTorch is not installed. Please install it first.')


__all__ = ['Tensor', 'FloatTensor', 'DoubleTensor', 'LongTensor']


class Tensor(torch.Tensor):
    '''
    !!! THIS CLASS CAN NOT BE INSTANTIATED, it is only used for type hinting!!!

    This class implements dunder method `__get_pydantic_core_schema__`.

    So you can use pydantic to validate torch tensors.

    Example:
    >>> import torch
    >>> from pydantic import validate_call
    >>> from sympydantic import Tensor

    >>> @validate_call
    ... def foo(arr: Tensor) -> None:
    ...     # !!! Tensor is `sympydantic.Tensor`
    ...     # not `torch.Tensor`
    ...     print(type(arr))
    ...     print(arr)

    >>> foo(torch.tensor([1, 2, 3]))  # right answer passed validation
    <class 'torch.Tensor'>
    tensor([1, 2, 3])

    >>> # wrong answer automatically cast to ndarray by pydantic
    >>> foo([4, 5, 6])
    <class 'torch.Tensor'>
    tensor([4, 5, 6])

    >>> # even a float scalar will be cast to a 0-dim int8 array by pydantic
    >>> foo(3.5)
    <class 'torch.Tensor'>
    tensor(3.5000)

    >>> # but very wrong answer will still raise PydanticValidationError
    >>> foo('aaa')
    Traceback (most recent call last):
        ...tracebacks...
    pydantic_core._pydantic_core.ValidationError: 1 validation error for foo
        ...tracebacks...
    '''
    _dtype: torch.dtype | None = None

    def __new__(cls, *args, **kwargs) -> NoReturn:
        raise NotImplementedError(
            'This class is only use for type hinting. '
            'If you want to create an instance, '
            'please use `torch.tensor()` or `torch.Tensor()` instead.'
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: type,  # source_type take the origin info of Generics,
        handlers: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_plain_validator_function(
            partial(cls._validate, source_type),
        )

    @classmethod
    def _validate(
        cls,
        source_type: type,
        value: Any,
        info: core_schema.ValidationInfo
    ) -> torch.Tensor:
        config = {} if info.config is None else info.config
        strict = config.get('strict', False)
        if strict and not isinstance(value, torch.Tensor):
            raise TypeError(
                f'value is not a torch.Tensor (type: {type(value)})'
            )
        if isinstance(value, torch.Tensor):
            if cls._dtype is not None and value.dtype != cls._dtype:
                value = value.to(cls._dtype)
            return value
        return torch.tensor(value).to(cls._dtype)


class FloatTensor(Tensor):
    _dtype = torch.float32


class DoubleTensor(Tensor):
    _dtype = torch.float64


class LongTensor(Tensor):
    _dtype = torch.int64
