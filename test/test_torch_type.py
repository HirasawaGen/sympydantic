import pytest
from pytest import mark
from pydantic import validate_call

try:
    import torch
except ImportError:
    pytest.skip("Numpy is not installed, This file was skipped.", allow_module_level=True)

from sympydantic import FloatTensor


@mark.parametrize('arg', [
    torch.randn(3, 4).int(),
])
@validate_call
def test_float_tensor(
    arg: FloatTensor,
): 
    assert arg.dtype == torch.float32