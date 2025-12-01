from typing import get_args, cast, Any
from functools import partial
import warnings

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

# 我希望最终发行版并不刚需numpy作为依赖，所以这里这样写了
try:
    import numpy as np
except ImportError:
    raise ImportError('Numpy is not installed, install numpy first.')

if np.__version__ < '1.20':
    raise ImportError('Your numpy version is too old, please upgrade to 1.20 or later.')

import numpy.typing as npt
# NOTE: _npt is temporally not used.
import numpy._typing as _npt  # noqa: F401


__all__ = ['NDArray']


class NDArray[T: np.generic](npt.NDArray[T]):  # type: ignore
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('This class is only used for type hinting, if you want to create an instance, use `np.array` or `np.ndarray` instead.')
    
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
    ) -> npt.NDArray[T]:
        config = {} if info.config is None else info.config
        strict = config.get('strict', False)
        if strict and not isinstance(value, np.ndarray):
            raise TypeError(f'value is not a numpy array: {value!r}')
        args = get_args(source_type)
        dtype = args[0] if len(args) else np.float64
        if strict and value.dtype != dtype:
            raise TypeError(f'value is not a numpy array of dtype {dtype}: {value!r}')
        if isinstance(value, np.ndarray):
            return value.astype(dtype)
        return np.array(value).astype(dtype)
    