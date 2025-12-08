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
        source_type: type,  # source_type will take the origin info of Generics,
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
            raise TypeError(f'value is not a torch.Tensor (type: {type(value)})')
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
    

