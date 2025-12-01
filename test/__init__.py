import sys
from pathlib import Path
from typing import Callable, Any
from functools import wraps

from pydantic import ValidationError

# This is a monkey patch
import _patch # noqa: F401
from metadatas.protocols import TensorLike


sys.path.append(str(Path(__file__).parent.parent.absolute()))
DEFAULT_TENSOR = 'numpy'

__all__ = ['should_raise','invalid_call', 'create_tensor', 'DEFAULT_TENSOR']


def should_raise(*exception_types: type[Exception]):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
                assert True, f'Exception type {exception_types.__name__} was not raised'
            except exception_types as e:
                print(e.__class__.__name__)
                print(f'Exception {e} was raised as expected')
        return wrapper
    return deco


def invalid_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
            assert False, 'ValidationError was not raised'
        except ValidationError as e:
            for error in e.errors():
                print()
                print()
                print(f'type: {error["type"]}')
                print(f'loc: {error["loc"]}')
                print(f'msg: {error["msg"]}')
    return wrapper


type _ProduceFunc = Callable[[tuple[int, ...], Any], TensorLike]

_registered: dict[str, _ProduceFunc] = {}

def _register(name: str):
    def deco(producer: _ProduceFunc):
        _registered[name] = producer
        return producer
    return deco

def create_tensor(shape: tuple[int, ...], dtype: Any = None, name: str = '') -> TensorLike:
    if name == '':
        name = DEFAULT_TENSOR
    if name not in _registered:
        raise ValueError(f'TensorLikeFactory: no producer registered under name {name}')
    return _registered[name](shape, dtype)
        

try:
    import numpy as np
    @_register('numpy')
    def numpy_tensor(shape: tuple[int, ...], dtype: Any = None) -> np.ndarray:
        if dtype is None:
            dtype = np.float32
        return np.zeros(shape, dtype=dtype)
except ImportError:
    print('numpy is not installed, numpy_tensor factory not registered')


try:
    import torch
    @_register('torch')
    def torch_tensor(shape: tuple[int, ...], dtype: Any = None) -> torch.Tensor:
        if dtype is None:
            dtype = torch.float32
        return torch.zeros(shape, dtype=dtype)
except ImportError:
    print('torch is not installed, torch_tensor factory not registered')