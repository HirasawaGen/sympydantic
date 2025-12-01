from . import create_tensor
from metadatas.protocols import TensorLike


def test_tensorlike_protocol() -> None:
    tensor_obj = create_tensor((2, 3))
    assert isinstance(tensor_obj, TensorLike)

