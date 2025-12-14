import warnings
from typing import Any, Final

from pydantic_core import core_schema

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


CPU: Final[Device] = Device(device='cpu')
'''
This metadata will cast your tensor to CPU device.

you can use it like: `Annotated[sympydantic.Tensor, sympydantic.CPU]`
'''

CUDA_: Final[tuple[Device]]
'''
This is a tuple of metadatas, each of which will cast your tensor
to a specific CUDA device.

size of this tuple is equal to the number of CUDA devices.

you can use it like: `Annotated[sympydantic.Tensor, sympydantic.CUDA_[2]]`
'''

CUDA: Final[Device | None]
'''
This metadata will cast your tensor to your first CUDA device.

you can use it like: `Annotated[sympydantic.Tensor, sympydantic.CUDA]`

If your cuda is not available, it will be None.
'''

if torch.cuda.is_available():
    _DEVICE_COUNT = torch.cuda.device_count()

    CUDA_ = tuple(
        Device(device=f'cuda:{i}')
        for i in range(_DEVICE_COUNT)
    )
    CUDA = CUDA_[0] if len(CUDA_) else None
else:
    CUDA_ = tuple()
