from typing import override, Any

from pydantic_core import PydanticCustomError

from . import _BaseCustomError



__all__ = [
    'DimensionConflictError',
    'ShapeValidatedError',
    'ShapeUnvalidatedError',
    'ShapeFormatError',
    'ShapeConflictError',
    'VarDimentionConflictError'
]


class DimensionConflictError(_BaseCustomError):
    '''
    If you provide a field as `arr: Annotated[TensorLike, tensorshape[3, 4]],
    But argument you provide is shape as [3, 4, 4],
    expected ndim is 2, but you provide 3.
    DimensionConflictError will raised.
    '''
    _error_type: str = 'dimension_conflict'
    _message_template: str = "The shape of this tensor-like is expected as {req_shape},"\
        "but you provide {orig_shape}, the ndim is {orig_ndim}, which is conflict on the number of dimensions."
    
    @override
    def __new__(cls, req_shape: tuple, orig_shape: tuple[int,...]):
        orig_ndim = len(orig_shape)
        return super().__new__(cls,
            {
                'req_shape': req_shape,
                'orig_shape': orig_shape,
                'orig_ndim': orig_ndim,
            }
        )


class ShapeValidatedError(_BaseCustomError):
    '''
    If you provide a field as `arr: Annotated[TensorLike, tensorshape[3, 4], tensorshape[9, 10]]`.
    The shape has been validated twice, which is not allowed in sympydantic.
    Then ShapeValidatedError will raised.
    '''
    _error_type: str = 'shape_validated'
    _message_template: str = 'tensor shape for {field_name} has already been validated'
    
    @override
    def __new__(cls, field_name: str):
        return super().__new__(cls, {'field_name': field_name})


class ShapeUnvalidatedError(_BaseCustomError):
    '''
    If you provide two field: `arr1: Annotated[TensorLike, tensorshape[...]], arr2: Annotated[TensorLike, tensorshape['*arrr1']]`.
    You hope `arr2` has same shape with `arr1`, but you have a spell error on `arrr1`
    `arrr1` has never been validated, sympydantic don't know shape of `arrr1`.
    Then ShapeUnvalidatedError will raised.
    '''
    _error_type: str = 'shape_unvalidated'
    _message_template: str = 'tensor shape for {field_name} has never been validated'
    
    @override
    def __new__(cls, field_name: str):
        return super().__new__(cls, {'field_name': field_name})

class ShapeFormatError(_BaseCustomError):
    '''
    If you provide a field: `arr: Annotated[TensorLike, tensorshape[3, ..., 4, ...]]`
    or `arr: Annotated[TensorLike, tensorshape[3.9, {'a': 13}]]` which is wrong format.
    ShapeFormatError will raised.
    '''
    _error_type: str = 'shape_format'
    _message_template: str = 'The tensor shape expression has a wrong format: {reason}'
    
    @override
    def __new__(cls, reason: str):
        return super().__new__(cls, {'reason': reason})
    
    

class ShapeConflictError(_BaseCustomError):
    '''
    If you provide a field as `arr: Annotated[TensorLike, tensorshape[3, 4]],
    But argument you provide is a wrong shape as [4, 3],
    They are same ndim, so DimensionConflictError will not raised.
    Instead ShapeConflictError will raised.
    '''
    _error_type: str ='shape_conflict'
    _message_template: str = 'dimension {dim} has length {orig_len}, expected {req_len}'
    
    @override
    def __new__(cls, dim: int, req_len: Any, orig_len: int):
        return super().__new__(cls, {
            'dim': dim,
            'req_len': req_len,
            'orig_len': orig_len
        })


class VarDimentionConflictError(_BaseCustomError):
    '''
    If you provide a field as `arr: Annotated[TensorLike, tensorshape[3, ..., 3]],
    But argument you provide is shape as [3],
    expected ndim is greater equal than 2, but you provide 1.
    VarDimentionConflictError will raised.
    '''
    _error_type: str = 'dimension_conflict'
    _message_template: str = "The shape of this tensor-like is expected as {req_shape}, which meas provided ndim should be at least {req_ndim},"\
        "but you provide {orig_shape}, the ndim is {orig_ndim}, which is less than the required number."
        
    @override
    def __new__(cls, req_shape: tuple, orig_shape: tuple[int,...]):
        orig_ndim = len(orig_shape)
        req_ndim = len(req_shape) - 1
        req_shape = str(req_shape).replace('Ellipsis', '...')
        return super().__new__(cls, {
            'orig_shape': orig_shape,
            'orig_ndim': orig_ndim,
            'req_shape': req_shape,
            'req_ndim': req_ndim,
        })

