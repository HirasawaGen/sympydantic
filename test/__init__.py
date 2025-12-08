from typing import Callable, Any
from typing import overload
from functools import wraps, partial

from pydantic import ValidationError

from sympydantic import TensorLike

__all__ = ['invalid_call', 'create_tensor', 'DEFAULT_TENSOR']
DEFAULT_TENSOR = 'numpy'



@overload
def invalid_call[**P, R](error_type: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    '''
    If the decorated function raises a ValidationError with the specified error_type,
    this decorator will not raise an AssertionError.
    If the decorated function runs without raising a ValidationError, this decorator will
    raise an AssertionError.
    >>> from annotated_types import Gt, Annotated
    
    >>> @invalid_call('greater_than')
    ... def foo(x: Annotated[int, Gt(0)]):
    ...     pass
    
    >>> foo(-1)  # raise ValidationError with type 'greater_than'
    ... type: greater_than
    ... loc: ('x',)
    ... msg: ...some message...
    
    >>> foo(1)
    ... AssertionError: ValidationError was not raised
    
    >>> foo('aaa')  # raise ValidationError, but not with type 'greater_than'
    ... AssertionError: ValidationError was not raised with type greater_than
    '''
    pass

@overload
def invalid_call[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    '''
    If the decorated function raises a ValidationError with the any error_type,
    this decorator will not raise an AssertionError.
    If the decorated function runs without raising a ValidationError, this decorator will
    raise an AssertionError.
    >>> from annotated_types import Gt, Annotated
    
    >>> @invalid_call('greater_than')
    ... def foo(x: Annotated[int, Gt(0)]):
    ...     pass
    
    >>> foo(-1)  # raise ValidationError with type 'greater_than'
    ... type: greater_than
    ... loc: ('x',)
    ... msg: ...some message...
    
    >>> foo(1)
    ... AssertionError: ValidationError was not raised
    
    >>> foo('aaa')  # raise ValidationError with type 'int_parsing'
    ... type: int_parsing
    ... loc: ('x',)
    ... msg: ...some message...
    '''
    pass


def _invalid_call[**P, R](func: Callable[P, R], error_type: str = ''):
    if error_type != '':
        error_type = error_type.lower().strip()
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        try:
            func(*args, **kwargs)
            assert False, 'ValidationError was not raised'
        except ValidationError as e:
            has_the_type = False
            for error in e.errors():
                if error_type != '' and error['type'] != error_type: continue
                has_the_type = True
                print()
                print()
                print(f'type: {error["type"]}')
                print(f'loc: {error["loc"]}')
                print(f'msg: {error["msg"]}')
            assert has_the_type, f'ValidationError was not raised with type {error_type}'
    return wrapper
    


def invalid_call[**P, R](arg: Callable[P, R] | str):
    if isinstance(arg, str):
        return partial(_invalid_call, error_type=arg)
    else:
        return _invalid_call(arg)


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