from typing import get_args, Any, NoReturn
from functools import partial

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

# 我希望最终发行版并不刚需numpy作为依赖，所以这里这样写了
# English is more international, right?
# The obove comment I wrote: "I hope the final release does not depend on numpy as a dependency, so I coded this way."
try:
    import numpy as np
except ImportError:
    raise ImportError('Numpy is not installed, install numpy first.')

if np.__version__ < '1.20':
    raise ImportError('Your numpy version is too old, please upgrade to 1.20 or later.')

import numpy.typing as npt
from numpy.typing import ArrayLike, DTypeLike, NBitBase
# NOTE: _npt is temporally not used.
import numpy._typing as _npt  # noqa: F401


__all__ = ['NDArray', 'ArrayLike', 'DTypeLike', 'NBitBase']


class NDArray[T: np.generic](npt.NDArray[T]):  # type: ignore
    '''
    !!! THIS CLASS CAN NOT BE INSTANTIATED, it is only used for type hinting!!!
    This class implements dunder method `__get_pydantic_core_schema__`.
    So you can use pydantic to validate numpy arrays.
    For example:
    >>> import numpy as np
    >>> from pydantic import validate_call
    
    >>> @validate_call
    ... def foo(arr: NDArray[np.int8]) -> None:
    ...     # !!! NDArray is `sympydantic.NDArray`
    ...     # not `numpy.typing.NDArray`
    ...     print(type(arr))
    ...     print(arr)
    
    >>> foo(np.array([1, 2, 3]))  # right answer passed validation
    <class 'numpy.ndarray'>
    [1 2 3]
    
    >>> foo([4, 5, 6])  # wrong answer automatically cast to ndarray by pydantic
    <class 'numpy.ndarray'>
    [4 5 6]
    
    >>> foo(3.5)  # even a float scalar will be cast to a 0-dim int8 array by pydantic
    <class 'numpy.ndarray'>
    3
    
    >>> foo('aaa')  # but very wrong answer will still raise PydanticValidationError
    Traceback (most recent call last):
        ...tracebacks...
    pydantic_core._pydantic_core.ValidationError: 1 validation error for foo
        ...tracebacks...
    '''
    def __new__(cls, *args, **kwargs) -> NoReturn:
        raise NotImplementedError(
            'This class is only used for type hinting, '
            'if you want to create an instance, '
            'use `np.array` or `np.ndarray` instead.'
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
    