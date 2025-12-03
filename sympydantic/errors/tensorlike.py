from typing import override


from pydantic_core import PydanticCustomError
from metadatas import TensorLike


class DimensionConflictError(PydanticCustomError):
    _error_type: str = 'dimension_conflict'
    _message_template: str = "The shape of this tensor-like is expected as {req_shape},"\
        "but you provide {orig_shape}, the ndim is {orig_ndim}, which is conflict on the number of dimensions."
    
    @override
    def __new__(cls, req_shape: tuple, orig_shape: tuple[int,...]):
        orig_ndim = len(orig_shape)
        return super().__new__(
            cls,
            cls._error_type,
            cls._message_template,
            {
                'req_shape': req_shape,
                'orig_shape': orig_shape,
                'orig_ndim': orig_ndim,
            }
        )


class ShapeConflictError(PydanticCustomError):
    '''
    same on ndim, but not equal on shape
    like: (2, 3) vs (2, 4)
    '''
    # TODO: implement this
    _error_type ='shape_conflict'
    _message_template: str = ''


