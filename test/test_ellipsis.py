from typing import Annotated

from pytest import mark
from pydantic import validate_call

from sympydantic import tensorshape, TensorLike

from . import create_tensor


@mark.parametrize("arr",[
    create_tensor((8, 1, 2, 3, 8)),  # right
    create_tensor((8, 1, 8)),  # bugs here. expected pass the test, but it fails: shape (8, 1, 8) is not compatible with required shape (8, Ellipsis, 8)
    create_tensor((8, 8)),
])
@validate_call
def test_int_right(
    # arr's shape is required to be (8, 8)
    # right answer provide, test pass
    arr: Annotated[TensorLike, tensorshape[8, ..., 8]]
) -> None:
    assert arr.shape[0] == 8
    assert arr.shape[-1] == 8