import warnings
from typing import Any

from pydantic_core import core_schema

# 我希望最终发行版并不刚需pytorch作为依赖，所以这里这样写了
# 整个工程不需要torch作为刚需的依赖，但是这个子模块需要
try:
    import torch
except ImportError:
    raise ImportError('PyTorch is not installed. Please install it first.')

from .basemetadata import MyBaseMetadata


__all__ = ['Device', 'CPU', 'CUDA', 'CUDA_']


class Device(MyBaseMetadata):
    def __init__(self, device: str | torch.device):
        self._device = torch.device(device)
    
    def _validate(
        self,
        value: Any,
        info: core_schema.ValidationInfo,
    ) -> Any:
        # 会被DuckType糊弄过去 但是可以降低性能开销
        if getattr(value, 'device', None) == self._device:
            return value
        if isinstance(value, torch.Tensor):
            return value.to(self._device)
        config = {} if info.config is None else info.config
        strict = config.get('strict', False)
        if strict:
            raise TypeError(f'Expected Tensor, but got {type(value).__name__}')
        warnings.warn(f'Expected Tensor, but got {type(value).__name__}')
        return value


    
CPU = Device(device='cpu')

if torch.cuda.is_available():
    _DEVICE_COUNT = torch.cuda.device_count()
            
    CUDA_ = tuple(
        Device(device=f'cuda:{i}')
        for i in range(_DEVICE_COUNT)
    )
    CUDA = CUDA_[0] if len(CUDA_) else None
else:
    CUDA_ = tuple()
    
