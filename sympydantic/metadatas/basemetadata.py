from abc import ABC, abstractmethod
from typing import Any, Self, Protocol
from functools import lru_cache

from annotated_types import BaseMetadata
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class MyBaseMetadata(BaseMetadata, ABC):
    def __get_pydantic_core_schema__(
        self,
        source_type: type,
        handlers: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_after_validator_function(
            self._validate,
            handlers(source_type),
        )

    @abstractmethod
    def _validate(
        self,
        value: Any,
        info: core_schema.ValidationInfo,
    ) -> Any: ...


class _SupportsGetitem[K, V](Protocol):
    def __getitem__(self, key: K) -> V: ...


class SubscriptableMetadata(MyBaseMetadata):
    @classmethod
    @lru_cache(maxsize=1)
    def subscriptable(cls) -> _SupportsGetitem[Any, Self]:
        class Subscriptable:
            def __getitem__(self, *keys: Any):
                return cls(*keys)

        Subscriptable.__name__ = f'{cls.__name__}Subscriptable'
        Subscriptable.__qualname__ = f'{cls.__qualname__}Subscriptable'
        Subscriptable.__module__ = cls.__module__
        return Subscriptable()
