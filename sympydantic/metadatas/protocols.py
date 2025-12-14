from __future__ import annotations

from typing import Protocol, Sized, Iterable, Sequence, Any
from typing import runtime_checkable

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


__all__ = ['TensorLike']


@runtime_checkable
class _SupportsCompare[T](Protocol):
    def __lt__(self, other: T) -> bool: pass
    def __le__(self, other: T) -> bool: pass
    def __gt__(self, other: T) -> bool: pass
    def __ge__(self, other: T) -> bool: pass


type _SupportsCompareNumber = _SupportsCompare[float] | _SupportsCompare[int]


class _SupportsCompute(Protocol):
    def __add__(self, other: Any) -> _SupportsCompute: pass
    def __sub__(self, other: Any) -> _SupportsCompute: pass
    def __mul__(self, other: Any) -> _SupportsCompute: pass
    def __truediv__(self, other: Any) -> _SupportsCompute: pass


@runtime_checkable
class TensorLike(Protocol, Sized, Iterable):
    '''
    protocol of numpy.typing.NDArray and torch.tensor etc.
    '''
    @property
    def dtype(self) -> Any: pass

    @property
    def shape(self) -> tuple[int, ...]: pass

    @property
    def ndim(self) -> int: pass

    def min(
        self,
    ) -> _SupportsCompute: pass

    def max(
        self,
    ) -> _SupportsCompute: pass

    def reshape(
        self,
        shape: Sequence[int],
    ) -> TensorLike: pass

    def __getitem__(
        self,
        *key: Any
    ) -> _SupportsCompute: pass

    def __add__(
        self,
        other: _SupportsCompute,
    ) -> _SupportsCompute: pass

    def __sub__(
        self,
        other: _SupportsCompute,
    ) -> _SupportsCompute: pass

    def __matmul__(
        self,
        other: _SupportsCompute,
    ) -> _SupportsCompute: pass


@classmethod  # type: ignore[misc]
def __get_pydantic_core_schema__(
    cls,
    source_type: type,  # source_type will take the origin info of Generics,
    handlers: GetCoreSchemaHandler,
) -> core_schema.CoreSchema:
    def _validate(value: Any):
        if not isinstance(value, cls):
            raise TypeError(f'value is not TensorLike (type: {type(value)})')
        return value
    return core_schema.no_info_plain_validator_function(
        _validate,
    )


TensorLike.__get_pydantic_core_schema__ =\
    __get_pydantic_core_schema__  # type: ignore[attr-defined]
