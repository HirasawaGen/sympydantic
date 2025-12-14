import warnings
from typing import Any, Final

from pydantic_core import core_schema

try:
    import torch
except ImportError:
    raise ImportError('PyTorch is not installed. Please install it first.')

from .basemetadata import MyBaseMetadata


__all__ = ['RequireGrad', 'WITH_GRAD', 'NO_GRAD']


class RequireGrad(MyBaseMetadata):
    def __init__(self, require_grad: bool = True):
        self._require_grad = require_grad

    def _validate(
        self,
        value: Any,
        info: core_schema.ValidationInfo,
    ) -> Any:
        # 这里把判断`requires_grad`取值的逻辑提前了
        # 虽然会被鸭子类型糊弄过去
        # 但是可以减少性能开销
        if getattr(value, 'requires_grad', None) == self._require_grad:
            return value
        config = {} if info.config is None else info.config
        strict = config.get('strict', False)
        if isinstance(value, torch.Tensor):
            if value.dtype not in {torch.float32, torch.float64}:
                if strict:
                    raise TypeError(
                        f'Expected float tensor, but got {value.dtype}'
                    )
                value = value.float()
            return value.requires_grad_(self._require_grad)
        if strict:
            raise TypeError(f'Expected Tensor, but got {type(value).__name__}')
        warnings.warn(f'Expected Tensor, but got {type(value).__name__}')
        return value

    def __invert__(self):
        return RequireGrad(not self._require_grad)


WITH_GRAD: Final[RequireGrad] = RequireGrad(True)
'''
This metadata will cast your tensor to float and set `requires_grad` to `True`
'''

NO_GRAD: Final[RequireGrad] = ~WITH_GRAD
'''
This metadata will cast your tensor to float and set `requires_grad` to `False`
'''
