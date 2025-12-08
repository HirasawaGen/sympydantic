from abc import ABC
from typing import override

from pydantic_core import PydanticCustomError


class _BaseCustomError(ABC, PydanticCustomError):
    # TODO: change all error to the subclass of this class 
    _error_type: str
    _message_template: str
    
    @override
    def __new__(cls, /, content: dict):
        return super().__new__(
            cls,
            cls._error_type,
            cls._message_template,
            content,
        )