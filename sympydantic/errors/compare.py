from typing import override, Any, final
from abc import ABC

from pydantic_core import PydanticCustomError


__all__ = [
    'DimensionGeError',
    'DimensionGtError',
    'DimensionLeError',
    'DimensionLtError'
]


class _BaseDimensionCompareError(PydanticCustomError, ABC):
    _error_type: str
    _template: str

    @override
    def __new__(cls, dimension: int, value: Any, provide: Any):
        return super().__new__(
            cls,
            cls._error_type,
            "The {dimension}-th dimension of"
            + "this tensor-like object should be "
            + cls._template
            + " {value}. You provide {provide}.",
            {
                'dimension': dimension,
                'value': value,
                'provide': provide
            }
        )


@final
class DimensionGeError(_BaseDimensionCompareError):
    _error_type: str = 'greater_than_equal'
    _template: str = 'greater than or equal to'


@final
class DimensionGtError(_BaseDimensionCompareError):
    _error_type: str = 'greater_than'
    _template: str = 'greater than'


@final
class DimensionLeError(_BaseDimensionCompareError):
    _error_type: str = 'less_than_equal'
    _template: str = 'less than or equal to'


@final
class DimensionLtError(_BaseDimensionCompareError):
    _error_type: str = 'less_than'
    _template: str = 'less than'
