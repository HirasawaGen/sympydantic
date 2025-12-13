'''
sympydantic
=====

Provides
  1. blablablah
  2. blublubluh
  3. etc.
'''
__author__ = 'HirasawaGen'
__repo__ = 'https://github.com/HirasawaGen/sympydantic'
__version__ = '0.1.0'


# This is monkey patchs
from . import _patch


__all__ = [
    'validate_call',
    'TensorLike',
    'tensorshape',
    'nrange',
]


from pydantic import validate_call
from .metadatas import TensorLike, tensorshape, nrange

try:
    from .dataschemas.numpy import *
    __all__.append('NDArray')
except ImportError:
    pass

try:
    from .dataschemas.torch import *
    __all__.append('Tensor')
except ImportError:
    pass


