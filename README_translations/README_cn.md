# sympydantic

一个集成了 SymPy、Pydantic、NumPy 和 PyTorch 的 Python 库，用于带符号数学验证的类型安全数值计算。

## 特性

- **符号验证**：利用 SymPy 的符号数学定义动态约束（例如张量形状关系 `2*X`）。
- **类型安全张量**：统一的 `TensorLike` 协议，支持 NumPy 数组和 PyTorch 张量的 Pydantic 验证。
- **上下文感知检查**：使用共享的 SymPy 符号命名空间进行跨参数验证（例如确保函数参数间 `y > x`）。
- **元数据验证器**：
  - `tensorshape`：使用符号表达式验证张量维度（例如 `tensorshape[X, 2*X]` 表示宽度为高度的2倍）
  - `nrange`：使用符号验证数值范围（例如 `nrange[X:Y]` 其中 X/Y 是动态值）
  - `Device`：强制张量设备放置（CPU/CUDA）
  - `RequireGrad`：控制 PyTorch 张量的梯度跟踪
- **可选依赖**：核心功能无需 NumPy/PyTorch；安装相关库后自动激活专用功能。

## 安装

```bash
# Core installation (SymPy + Pydantic)
pip install git+https://github.com/HirasawaGen/sympydantic.git
```

Linux 系统上的 CUDA 支持会自动使用 PyTorch 的 CUDA 12.8 索引源。

## 快速开始

### 符号张量形状验证

```python
from typing import Annotated
from pydantic import validate_call
from sympy.abc import X  # 导入 SymPy 符号
from sympydantic import tensorshape, TensorLike

@validate_call
def process_matrix(
    # 验证 2D 张量，其中宽度 = 3 * 高度（使用符号 X）
    matrix: Annotated[TensorLike, tensorshape[X, 3*X]]
) -> tuple[int, int]:
    return matrix.shape

# 有效：形状 (2, 6) 满足 6 = 3*2
process_matrix([[1,2,3,4,5,6], [7,8,9,10,11,12]])

# 无效：形状 (2, 4) 不满足 4 ≠ 3*2（触发 ValidationError）
process_matrix([[1,2], [3,4]])
```

### 跨参数符号约束

```python
from typing import Annotated
from pydantic import validate_call
from sympy.abc import X, Y
from sympydantic import nrange

@validate_call
def constrained_values(
    x: Annotated[int, X],
    y: Annotated[int, Y],
    # z 必须在 x 和 x+y 之间（动态范围）
    z: Annotated[int, nrange[X:X+Y]]
) -> tuple[int, int, int]:
    return x, y, z

# 有效：5 在 3 和 3+4=7 之间
constrained_values(3, 4, 5)

# 无效：8 不在 3 和 7 之间（触发 ValidationError）
constrained_values(3, 4, 8)
```

### PyTorch 设备和梯度控制

```python
from typing import Annotated
import torch
from sympydantic import Device, RequireGrad, TensorLike

@validate_call
def prepare_model_input(
    # 确保张量在 CUDA 上且启用梯度
    features: Annotated[TensorLike, Device('cuda'), RequireGrad(True)]
) -> TensorLike:
    return features

# 创建无梯度的 CPU 张量
cpu_tensor = torch.tensor([1.0, 2.0])

# 验证过程自动将张量移至 CUDA 并启用梯度
gpu_tensor = prepare_model_input(cpu_tensor)
print(gpu_tensor.device)        # cuda:0
print(gpu_tensor.requires_grad) # True
```

## 核心组件

### 数据结构
- `NDArray`：带类型注解的 NumPy 数组，支持 Pydantic 验证（位于 `dataschemas/numpy.py`）。
- `Tensor`：带类型注解的 PyTorch 张量，支持设备/梯度控制（位于 `dataschemas/torch.py`）。

### 元数据验证器
- `tensorshape`：使用 SymPy 表达式验证维度（例如 `tensorshape[..., X, 2*X]`）。
- `nrange`：使用符号边界验证数值范围（例如 `nrange[X:Y]`）。
- `Device`：强制张量设备（CPU/CUDA）并支持自动转换。
- `RequireGrad`：控制 PyTorch 张量的梯度跟踪。

### 协议
- `TensorLike`：跨 NumPy 和 PyTorch 的数组/张量统一接口。
