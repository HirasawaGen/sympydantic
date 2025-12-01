# This is monkey patchs
from . import _patch

from pydantic import validate_call
from .metadatas import TensorLike, tensorshape, nrange