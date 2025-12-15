<p align="right">
  <a href="./README.md">🇬🇧 English</a> |
  <a href="./README_zh.md">🇨🇳 简体中文</a> |
  <a href="./README_ja.md">🇯🇵 日本語</a>
</p>

![Pydantic](https://img.shields.io/badge/Pydantic-%E2%89%A5%202.12.4-0e7fc0?logo=pydantic&logoColor=white)
![SymPy](https://img.shields.io/badge/SymPy-%E2%89%A5%201.14.0-3f7aa0?logo=sympy&logoColor=white)
![NumPy-optional](https://img.shields.io/badge/NumPy-optional-9c7c4f?logo=numpy)
![PyTorch-optional](https://img.shields.io/badge/PyTorch-optional-ee4c2c?logo=PyTorch&logoColor=white)

# Sympydantic Project Guide

## Project Background

Have you ever met the following situations 🧐?  
You trained a CNN for a whole afternoon, and finally—just because you **forgot to transpose a matrix**—the columns of the former tensor did not match the rows of the latter one. The program crashed and **all efforts were wiped out** in a second. 😭

Or… your **colleague** wrote a utility function that returns a `numpy` array, but the type hint says `np.int8`. Innocent you used one of its elements as a **list index**, only to find out the real dtype is `np.float64`! 🙀

Or in some **reinforcement-learning** environment, the state space is actually `tuple[int]`, yet the original developer **forgot to add type annotations**. People naturally treat it as a `numpy` array and happily call `arr.argmax()`—boom, red errors everywhere 😨.

You might be a **senior technical consultant** in a big company, or a Ph.D. student from a **top university**. Complicated **mathematical formulas** are nothing to you—you can handle them with ease. But here there is no `unsqueeze`, there you forgot a `reshape`. These seemingly trivial mistakes drive you crazy! 🤬

As the not-yet-**famous open-source developer** HirasawaGen once said:  
> “In deep learning, once the **shape** problems are solved, half of all problems disappear.” 😎

Try `sympydantic`! By combining the well-known Python libraries `pydantic` and `sympy`, it **pre-validates** the shapes of `numpy` or `torch` tensors **before** the function body is executed.  
Let the error that would have popped up while you were eating McDonald’s 🍔🍟 and watching anime emerge **seconds** after you hit the Run button ▶, so you can fix it **calmly and confidently**!

## Dependencies

- Python ≥ 3.12  
- Pydantic (data-validation library) ≥ 2.12.4  
- SymPy (symbolic mathematics) ≥ 1.14.0  

> Note: `sympydantic` is still under active development; therefore only **Python 3.12+** is supported for the moment 😩. Support for earlier versions will be added gradually.

## Core Features

### Automatic Shape Validation

#### Demo 1

The library’s name itself reveals the idea: merge `sympy` with `pydantic`.  
```python
from typing import Annotated
import numpy as np
import sympy as sp
from pydantic import validate_call

from sympydantic import TensorLike   # Protocol satisfied by both torch.Tensor and numpy.ndarray
from sympydantic import tensorshape  # Metadata used to describe tensor shapes

X = sp.symbols('X')

@validate_call
def foo(
    arg: Annotated[TensorLike, tensorshape[2, X, X + 2]],
) -> None:
    # !! If you do NOT see arg.shape printed, pydantic intercepted the call
    #    before the real function body was reached.
    print(arg.shape)
    assert arg.shape[0] == 2
    assert arg.shape[1] + 2 == arg.shape[2]

if __name__ == '__main__':
    arg1 = np.random.rand(2, 3, 5)
    arg2 = np.random.rand(1, 3, 5)  # 1st dim should be 2
    arg3 = np.random.rand(2, 3, 4)  # 2nd vs. 3rd dim conflict

    foo(arg1)  # OK

    try:
        foo(arg2)
    except Exception as e:
        print(e)  # dimension 0 has length 1, expected 2 (int)

    try:
        foo(arg3)
    except Exception as e:
        print(e)  # The expression 'X + 2' is solved as 5, which conflicts with the provided value 4.

''' Terminal Output:
(2, 3, 5)
1 validation error for foo
0
  dimension 0 has length 1, expected 2 (int) 
  [type=shape_conflict, input_value=array([[[0.22684143, 0.50...66766634, 0.46905961]]]), input_type=ndarray]
1 validation error for foo
0
  The expression 'X + 2' is solved as 5, which is conflict with the provided value 4. 
  [type=expr_conflict, input_value=array([[[0.59563589, 0.36...08101385, 0.58254737]]]), input_type=ndarray]
'''

```

`arg1` has shape `(2, 3, 5)`, satisfying the constraint `(2, X, X+2)`.  
`arg2` and `arg3` violate the constraint, so the calls are rejected **before** entering the function body.

#### Demo 2

If you don’t want to declare a `sympy.Symbol` explicitly, use `TypeVar`:
```python
from typing import Annotated
import numpy as np
from pydantic import validate_call

from sympydantic import TensorLike, tensorshape

@validate_call
def foo[X](
    arg: Annotated[TensorLike, tensorshape[X, X]],
) -> None:
    print(arg.shape)
    assert arg.shape[0] == arg.shape[1]

if __name__ == '__main__':
    arg1 = np.random.rand(3, 3)
    arg2 = np.random.rand(3, 4)

    foo(arg1)  # OK

    try:
        foo(arg2)
    except Exception as e:
        print(e)  # The symbol 'X' is already set to 3. You provided a conflicting value 4.

''' Terminal Output:
(3, 3)
1 validation error for foo
0
  The symbol 'X' is already set to 3. you provide a conflict value 4.
  [type=symbol_redefined, input_value=array([[0.40639904, 0.541....92482645, 0.0740373 ]]), input_type=ndarray]
'''

```

#### Demo 3

`TypeVar` does not support arithmetic. If you still dislike importing `sympy`, write  
`tensorshape['X', 'X+1']` directly. You can also mix numbers, strings, and `slice` objects:
```python
from typing import Annotated
import numpy as np
from sympy.abc import X, Y
from pydantic import validate_call

from sympydantic import TensorLike, tensorshape

@validate_call
def foo(
    value_Y: Annotated[int, Y],              # binds the value of symbol Y
    arg1: Annotated[TensorLike, tensorshape[X, X:10, '*']],
    arg2: Annotated[TensorLike, tensorshape[..., '2 * Y - 1']],
) -> None:
    print(arg1.shape)
    print(arg2.shape)
    _solve_X = arg1.shape[0]
    assert _solve_X <= arg1.shape[1] < 10   # slice mixed with numbers
    assert arg2.shape[-1] == 2 * value_Y - 1  # only the last dim is checked
    # pydantic verifies arg1.ndim == 3, but does NOT care arg2.ndim
```

Remarks  
- A dimension annotated as `'*'` is **completely ignored**; its value is neither validated nor stored.  
- Normally, `sympydantic` first checks **ndim**. For example, expected shape `(X, X:10, Y)` will reject `(1,2,3,4,5)` immediately.  
- Ellipsis `...` means “I only care about the prefix and/or suffix”.  
  Example: `(1, ..., X+2)` validates the first and the last dims; anything in between is free.  
- If you **do** want exactly five dimensions while validating only head and tail, write `(X, '*', '*', '*', 2*X)`.

### Automatic Type Conversion

#### Demo 4

The previous demos used `TensorLike`, which **never** converts your data.  
To enable auto-conversion, do:
```python
from typing import Annotated
import numpy as np
import torch
from pydantic import validate_call

from sympydantic import TensorLike, Tensor, NDArray
from sympydantic.metadatas.device import CUDA   # raises error if CUDA-torch unavailable

@validate_call
def foo(
    original_arr: TensorLike,                       # validation only
    numpy_arr: Annotated[NDArray[np.bool], 'meta'], # converts to numpy + bool
    torch_arr: Annotated[Tensor, CUDA],             # converts to torch + cuda
) -> None:
    print(original_arr)
    print(numpy_arr)
    print(torch_arr)

if __name__ == '__main__':
    numpy_arr = np.random.rand(3).astype(np.float64)
    foo(numpy_arr, numpy_arr, numpy_arr)

    torch_arr = torch.rand(3)
    foo(torch_arr, numpy_arr, torch_arr)

''' Terminal Output:
[0.71413676 0.09614301 0.04009426]
[ True  True  True]
tensor([0.7141, 0.0961, 0.0401], device='cuda:0', dtype=torch.float64)
tensor([0.1790, 0.4157, 0.8533])
[ True  True  True]
tensor([0.1790, 0.4157, 0.8533], device='cuda:0')
'''

```

#### Demo 5

You can even pass a plain Python list or a scalar:
```python
from typing import Annotated
import numpy as np
import torch
from pydantic import validate_call

from sympydantic import TensorLike, Tensor, NDArray
from sympydantic.metadatas.device import CUDA

@validate_call
def foo(
    numpy_arr: Annotated[NDArray[np.bool], 'meta'],
    torch_arr: Annotated[Tensor, CUDA],
) -> None:
    print(numpy_arr)
    print(torch_arr)

if __name__ == '__main__':
    arr = [1, 2, 3]
    foo(arr, arr)
    foo(3, 9)


''' Terminal Output:
[ True  True  True]
tensor([1, 2, 3], device='cuda:0')
True
tensor(9, device='cuda:0')
'''

```
No more worries about whether that RL environment returns a `tuple` or an `ndarray`!

## How To Use?

if you use `pip`:

```cmd
pip install https://github.com/HirasawaGen/sympydantic.git
```

or `uv`:

```cmd
uv add https://github.com/HirasawaGen/sympydantic.git
```

## TODOs

- Auto-squeeze: e.g. expected shape `(1, 3, 4, 5)` accepts `(3, 1, 4, 5, 1)` and reshapes it automatically.  
- Broadcasting support: e.g. annotate `(3, 4, 4)` and pass a scalar, then broadcast it to `(3, 4, 4)`.  
- Support Python 3.8–3.11.  
- Support metadatas which validate whether a matrix is symmetric, positive-definite, and rank-deficient, etc. 
- dynamic validation is wasting performance. Maybe I should to code a mypy plugins to optimize it.