# sympydantic

A Python library integrating SymPy, Pydantic, NumPy, and PyTorch for type-safe numerical computing with symbolic mathematics validation.

## Features

- **Symbolic Validation**: Leverage SymPy's symbolic mathematics to define dynamic constraints (e.g., tensor shape relationships like `2*X`).
- **Type-Safe Tensors**: Unified `TensorLike` protocol for NumPy arrays and PyTorch tensors with Pydantic validation.
- **Context-Aware Checks**: Cross-parameter validation using shared SymPy symbol namespaces (e.g., ensure `y > x` across function arguments).
- **Metadata Validators**: 
  - `tensorshape`: Validate tensor dimensions with symbolic expressions (e.g., `tensorshape[X, 2*X]` for square matrices scaled by 2)
  - `nrange`: Validate numerical ranges using symbols (e.g., `nrange[X:Y]` where X/Y are dynamic values)
  - `Device`: Enforce tensor device placement (CPU/CUDA)
  - `RequireGrad`: Control gradient tracking for PyTorch tensors
- **Optional Dependencies**: Core functionality works without NumPy/PyTorch; specialized features activate when libraries are installed.

## Installation

```bash
# Core installation (SymPy + Pydantic)
pip install git+https://github.com/HirasawaGen/sympydantic.git
```

For CUDA support on Linux, the installation automatically uses PyTorch's CUDA 12.8 index.

## Quick Start

### Symbolic Tensor Shape Validation

```python
from typing import Annotated
from pydantic import validate_call
from sympy.abc import X  # Import SymPy symbols
from sympydantic import tensorshape, TensorLike

@validate_call
def process_matrix(
    # Validate a 2D tensor where width = 3 * height (using symbol X)
    matrix: Annotated[TensorLike, tensorshape[X, 3*X]]
) -> tuple[int, int]:
    return matrix.shape

# Valid: Shape (2, 6) where 6 = 3*2
process_matrix([[1,2,3,4,5,6], [7,8,9,10,11,12]])

# Invalid: Shape (2, 4) where 4 ≠ 3*2 (raises ValidationError)
process_matrix([[1,2], [3,4]])
```

### Cross-Parameter Symbol Constraints

```python
from typing import Annotated
from pydantic import validate_call
from sympy.abc import X, Y
from sympydantic import nrange

@validate_call
def constrained_values(
    x: Annotated[int, X],
    y: Annotated[int, Y],
    # z must be between x and x+y (dynamic range)
    z: Annotated[int, nrange[X:X+Y]]
) -> tuple[int, int, int]:
    return x, y, z

# Valid: 5 is between 3 and 3+4=7
constrained_values(3, 4, 5)

# Invalid: 8 is not between 3 and 7 (raises ValidationError)
constrained_values(3, 4, 8)
```

### Device and Gradient Control for PyTorch

```python
from typing import Annotated
import torch
from sympydantic import Device, RequireGrad, TensorLike

@validate_call
def prepare_model_input(
    # Ensure tensor is on CUDA with gradients enabled
    features: Annotated[TensorLike, Device('cuda'), RequireGrad(True)]
) -> TensorLike:
    return features

# Create CPU tensor without gradients
cpu_tensor = torch.tensor([1.0, 2.0])

# Validation automatically moves to CUDA and enables gradients
gpu_tensor = prepare_model_input(cpu_tensor)
print(gpu_tensor.device)        # cuda:0
print(gpu_tensor.requires_grad) # True
```

## Core Components

### Data Structures
- `NDArray`: Type-annotated NumPy array with Pydantic validation (in `dataschemas/numpy.py`).
- `Tensor`: Type-annotated PyTorch tensor with device/gradient support (in `dataschemas/torch.py`).

### Metadata Validators
- `tensorshape`: Validate dimensions using SymPy expressions (e.g., `tensorshape[..., X, 2*X]`).
- `nrange`: Validate numerical ranges with symbolic bounds (e.g., `nrange[X:Y]`).
- `Device`: Enforce tensor device (CPU/CUDA) with automatic conversion.
- `RequireGrad`: Control gradient tracking for PyTorch tensors.

### Protocols
- `TensorLike`: Unified interface for array/tensor objects across NumPy and PyTorch.
