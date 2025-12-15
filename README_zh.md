<p align="right">
  <a href="./README.md">🇬🇧 English</a> |
  <a href="./README_zh.md">🇨🇳 简体中文</a> |
  <a href="./README_ja.md">🇯🇵 日本語</a>
</p>

![Pydantic](https://img.shields.io/badge/Pydantic-%E2%89%A5%202.12.4-0e7fc0?logo=pydantic&logoColor=white)
![SymPy](https://img.shields.io/badge/SymPy-%E2%89%A5%201.14.0-3f7aa0?logo=sympy&logoColor=white)
![NumPy-optional](https://img.shields.io/badge/NumPy-optional-9c7c4f?logo=numpy)
![PyTorch-optional](https://img.shields.io/badge/PyTorch-optional-ee4c2c?logo=PyTorch&logoColor=white)

# Sympydantic项目指南

## 项目背景

你是否遇到过这样的情况🧐？CNN网络训练了一个下午，最后因为一个**矩阵忘记转置**的原因，导致前一矩阵的列数不等于后一矩阵的行数，然后程序报错，**前功尽弃**？😭

或者是……你的**同事**设计的工具函数，返回的一个`numpy`数组，他类型标注是`np.int8`，于是无辜的你把他的某个元素值当作**列表索引**使用，结果他返回的是个`np.float64`！🙀

或者是某个**强化学习**环境，返回的状态空间是`tuple[int]`，但是源码开发者没写**类型标注**，那开发者总是会容易把他当成返回`numpy`数组处理对吧，然后一个`arr.argmax()`，咦~哐哐标红😨

你，可能是某大厂的**高级技术顾问**，可能是**名牌大学**的博士研究生，复杂的**数学公式**难不倒你，对你来说完全是信手拈来。但是这里没有`unsqueeze`，那里忘记`reshape`，这些明明很简单，但却又很容易出错的问题却搞得你想骂人！🤬

正如`GitHub`开源社区某三流不**知名程序员HirasawaGen**所说：
> “做深度学习任务时，**形状**的问题解决了，那么所有的问题就解决了一半。”😎

使用`sympydantic`吧！结合知名的`python`第三方工具`pydantic`和`sympy`，在函数运行之前提前对`numpy`数组或者`torch`数组的形状进行约束，让几个小时后你**边看动漫边吃麦当劳🍔🍟**的时候才报的错，在你点下**运行键▶**的几秒后就报了出来，让你可以**从从容容游刃有余**地解决！

## 项目依赖

- Python >= 3.12

- Pydantic: 数据验证库 (版本 >= 2.12.4)

- SymPy: 符号数学库 (版本 >= 1.14.0)

> 注：由于sympydantic仍在开发阶段，目前暂时只支持python 3.12+😩，后续会慢慢开发的

## 核心功能

### 自动形状验证

#### demo 1

首先，之所以起这个名字，肯定是因为`sympydantic`可以结合`sympy`与`pydantic`二者的优势，例如下面这个demo：

```python
from typing import Annotated  # 引入 Annotated 用来为类型注解标注元数据
# from typing_extensions import Annotated

import numpy as np
import sympy as sp
from pydantic import validate_call

from sympydantic import TensorLike  # TensorLike是一个协议，torch.Tensor与numpy.ndarray都满足这个协议
from sympydantic import tensorshape  # tensorshape是一个元数据，用来描述张量的形状


X = sp.symbols('X')


@validate_call
def foo(
    arg: Annotated[TensorLike, tensorshape[2, X, X+2]],
) -> None:
    # !! 如果执行foo时，没有输出arg.shape，说明还没等执行真正的函数内容，pydantic就拦截了你的调用
    print(arg.shape)
    assert arg.shape[0] == 2
    assert arg.shape[1] + 2 == arg.shape[2]


if __name__ == '__main__':
    arg1 = np.random.rand(2, 3, 5)
    arg2 = np.random.rand(1, 3, 5)  # 第一个维度应该是2
    arg3 = np.random.rand(2, 3, 4)  # 第二个维度与第三个维度未满足不等式
    
    foo(arg1)  # 正确
    
    try:
        foo(arg2)
    except Exception as e:
        print(e)  # dimension 0 has length 1, expected 2 (int) 
            
    try:
        foo(arg3)  # The expression 'X + 2' is solved as 5, which is conflict with the provided value 4.
    except Exception as e:
        print(e)
            

''' 终端输出:
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

注意，`arg1`的形状为`(2, 3, 5)`，满足约束`(2, X, X+2)`，即第一个维度为2，第二个维度比第三个维度小2，于是，`foo(arg1)`里面那行`print`正常执行。`arg2`的形状为`(1, 3, 5)`，不满足要求，因此未通过验证，`foo(arg2)`在执行之前就抛出异常。`arg3`同理。

#### demo 2

当然如果你不喜欢显式地声明一个sympy.Symbol对象，那也是可以的！使用TypeVar来实现。

```python
from typing import Annotated  # 引入 Annotated 用来为类型注解标注元数据
# from typing_extensions import Annotated

import numpy as np
from pydantic import validate_call

from sympydantic import TensorLike  # TensorLike是一个协议，torch.Tensor与numpy.ndarray都满足这个协议
from sympydantic import tensorshape  # tensorshape是一个元数据，用来描述张量的形状


@validate_call
def foo[X](
    # 尽管X其实是TypeVar，但是sympydantic会把它转为同名的sympy
    # 因此无需担心声明太多sympy对象污染您的命名空间
    arg: Annotated[TensorLike, tensorshape[X, X]],
) -> None:
    # !! 如果执行foo时，没有输出arg.shape，说明还没等执行真正的函数内容，pydantic就拦截了你的行为
    print(arg.shape)
    assert arg.shape[0] == arg.shape[1]


if __name__ == '__main__':
    arg1 = np.random.rand(3, 3)
    arg2 = np.random.rand(3, 4)
    
    foo(arg1)  # 正确
    
    try:
        foo(arg2)
    except Exception as e:
        print(e)  # The symbol 'X' is already set to 3. you provide a conflict value 4.           

''' 终端输出:
(3, 3)
1 validation error for foo
0
  The symbol 'X' is already set to 3. you provide a conflict value 4.
  [type=symbol_redefined, input_value=array([[0.40639904, 0.541....92482645, 0.0740373 ]]), input_type=ndarray]
'''

```

#### demo 3

不过可惜的是，`TypeVar`并不支持加减乘除等操作，那要是你还是不想导入sympy，使用`tensorshape['X', 'X+1']`也是可以的。

另外，使用`slice`对象，使用数字与字母混合等也是可以，

```python
from typing import Annotated  # 引入 Annotated 用来为类型注解标注元数据
# from typing_extensions import Annotated

import numpy as np
from sympy.abc import X, Y
from pydantic import validate_call

from sympydantic import TensorLike
from sympydantic import tensorshape


@validate_call
def foo(
    value_Y: Annotated[int, Y],   # 标注元数据，表示Symbol对象Y的值被绑定给了参数value_Y
    arg1: Annotated[TensorLike, tensorshape[X, X:10, '*']],
    arg2: Annotated[TensorLike, tensorshape[..., '2 * Y - 1']],
) -> None:
    print(arg1.shape)
    print(arg2.shape)
    _solve_X = arg1.shape[0]
    assert _solve_X <= arg1.shape[1] < 10  # (X:10) 使用slice，并且混搭了sympy与数字
    assert arg2.shape[-1] == 2 * value_Y - 1  # 使用Ellipsis，表明对一共有几维不进行验证，只验证最后一个维是不是满足'2 * Y - 1'

```

首先，若某个维度被标注为了字符串`*`，意思就是这个维度**不进行任何验证**，也不会把他的值存储给任何一个symbol。

另外，正常来说，`sympydantic`是会先验证维度数是否满足约束，例如若要求的形状是`(X, X:10, Y)`，而传入的形状是`(1, 2, 3, 4, 5)`，那首先连维度数都不匹配，就像正方形和三角锥，没有必要比较，就直接抛异常了。

但是`(1, ...)`，`(1, 2, 3, ..., X+2)`这种形式意思是，我要求你前几维度如何如何，后几维度如何如何，中间有几个我完全不关心。就像是我要这个形状是有棱有角的，只要整体轮廓符合约束即可，中间维度是三角形还是三棱锥都不影响。

那如果你非要让他一共有五维，并且只验证头尾，那就使用`(X, '*', '*', '*', 2*X)`

### 自动类型转换

#### demo 4

上面的几个`demo`中，`Annotated`里面都是`TensorLike`，他不会擅自自动转换你的数据类型，如果你想要使用自动类型转换，可以考虑下面这个例子：

```python
from typing import Annotated
# from typing_extensions import Annotated

import numpy as np
import torch

from pydantic import validate_call

from sympydantic import TensorLike
from sympydantic import Tensor  # 未安装torch则会报错
from sympydantic import NDArray  # 未安装numpy则会报错
from sympydantic.metadatas.device import CUDA  # 未安装cuda版本torch则会报错


@validate_call
def foo(
    original_arr: TensorLike,
    numpy_arr: Annotated[NDArray[np.bool], '这是一个用来充数的元数据'],
    torch_arr: Annotated[Tensor, CUDA],
) -> None:
    print(original_arr)  # 标注为TensorLike则不进行转换，只进行验证
    print(numpy_arr)  # 传入的数组自动转换为numpy.ndarray，并将dtype也转为np.bool
    print(torch_arr)  # 传入的数组自动转换为torch.Tensor，并将device转为CUDA
    
    
if __name__ == '__main__':
    numpy_arr = np.random.rand(3).astype(np.float64)
    foo(numpy_arr, numpy_arr, numpy_arr)
    torch_arr = torch.rand(3)
    foo(torch_arr, numpy_arr, torch_arr)

''' 终端输出：
[0.71413676 0.09614301 0.04009426]
[ True  True  True]
tensor([0.7141, 0.0961, 0.0401], device='cuda:0', dtype=torch.float64)
tensor([0.1790, 0.4157, 0.8533])
[ True  True  True]
tensor([0.1790, 0.4157, 0.8533], device='cuda:0')
'''
```

#### demo 5

甚至还能这样玩：

```python
from typing import Annotated
# from typing_extensions import Annotated

import numpy as np
import torch

from pydantic import validate_call

from sympydantic import TensorLike, Tensor, NDArray
from sympydantic.metadatas.device import CUDA


@validate_call
def foo(
    numpy_arr: Annotated[NDArray[np.bool], 'This is a meta data'],
    torch_arr: Annotated[Tensor, CUDA],
) -> None:
    print(numpy_arr)
    print(torch_arr)
    
    
if __name__ == '__main__':
    arr = [1, 2, 3]
    foo(arr, arr)
    foo(3, 9)

''' 终端输出：
[ True  True  True]
tensor([1, 2, 3], device='cuda:0')
True
tensor(9, device='cuda:0')
'''
```

这样你就不用担心某个**强化学习**环境到底给你`tuple`还是`ndarray`了！

## 如何使用

如果你使用`pip`:

```cmd
pip install https://github.com/HirasawaGen/sympydantic.git
```

或者你使用`uv`:

```cmd
uv add https://github.com/HirasawaGen/sympydantic.git
```

## TODOs

- 也许可以考虑把**squeeze**做进去，例如标注的形状是`(1, 3, 4, 5)`，传入的形状是`(3, 1, 4, 5, 1)`，就自动`resize`成需要的形状。
- 对广播的支持，例如标注为`(3, 4, 4)`，传入了标量，则把该标量广播到`(3, 4, 4)`
- 对python3.8到3.11这几个版本的支持。
- 支持一些可以验证矩阵是否对称，是否正定，是否满秩等的元数据。
- 动态验证浪费性能，也许我可以考虑编写一个mypy插件来优化这个问题。
