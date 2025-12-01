# sympydantic 项目指南

## 项目概述
sympydantic 是一个结合了 SymPy、Pydantic、NumPy 和 PyTorch 的 Python 项目，旨在提供类型安全的数值计算和模型验证功能。该项目主要用于创建与 NumPy 数组和 PyTorch 张量兼容的 Pydantic 模型，并提供自定义的验证逻辑。项目设计为可选依赖结构，核心功能不强制要求 numpy 和 torch 依赖。

## 项目结构
```
sympydantic/
├── _pyproject.toml      # 实际部署时的配置文件（无numpy/torch依赖）
├── pyproject.toml       # 测试环节使用的配置文件（含numpy/torch依赖）
├── README.md            # 项目说明文件
├── IFLOW.md             # 项目开发指南
├── .python-version      # Python 版本指定
├── .gitignore           # Git 忽略配置
├── uv.lock              # uv 锁定文件
├── _patch/              # 猴子补丁目录
│   ├── __init__.py
│   ├── pydantic.py      # Pydantic 相关补丁
│   └── sympy.py         # SymPy 表达式验证补丁
├── dataschemas/         # 数据架构定义
│   ├── __init__.py
│   ├── numpy.py         # NumPy 数组架构
│   └── torch.py         # PyTorch 张量架构
├── metadatas/           # 元数据定义
│   ├── __init__.py
│   ├── basemetadata.py  # 基础元数据类定义
│   ├── device.py        # 设备元数据（CPU/CUDA）
│   ├── grad.py          # 梯度元数据
│   ├── nrange.py        # 数字范围验证元数据
│   ├── protocols.py     # 协议定义
│   └── tensorshape.py   # 张量形状元数据
└── test/                # 测试文件
    ├── __init__.py
    ├── test_cast_numpy.py
    ├── test_demo.py
    ├── test_nrange.py
    ├── test_patchs.py
    ├── test_shape.py
    ├── test_sympy.py
    ├── test_tensorlike_protocol.py
    ├── test_timeit.py
    ├── test_typealias.py
    └── test_typevartuple.py
```

## 技术栈与依赖

### 核心依赖（部署版本，基于 _pyproject.toml）：
- Python >= 3.12
- Pydantic: 数据验证和设置管理库 (包含 mypy 支持) (版本 >= 2.12.4)
- SymPy: 符号数学库 (版本 >= 1.14.0)
- MyPy: 静态类型检查工具 (版本 >= 1.18.2)
- PyTest: 测试框架 (版本 >= 9.0.1)
- Annotated Types: 类型注解增强

### 可选依赖（测试版本，基于 pyproject.toml）：
- NumPy: 数值计算库 (版本 >= 2.3.5)
- PyTorch: 深度学习框架 (GPU 支持) (版本 >= 2.9.1)
- MyPy: 静态类型检查工具 (版本 >= 1.18.2)

项目采用可选依赖策略，核心功能不强制要求 numpy 和 torch，但在测试或使用特定功能时需要这些依赖。

### 配置特性
- **PyTorch CUDA 支持**: 配置了专门的 PyTorch CUDA 12.8 索引源，自动在 Linux 平台启用 CUDA 支持
- **MyPy 插件**: 集成 Pydantic MyPy 插件，提供更精确的类型检查
- **PyTest 配置**: 启用详细输出和日志记录，便于调试

## 核心功能

### NDArray 类型
- 在 `dataschemas/numpy.py` 中定义
- 自定义的 `NDArray[T: np.generic]` 类型，继承自 `numpy.typing.NDArray[T]`
- 支持条件导入 numpy（如果未安装则抛出 ImportError）
- 提供了 `__get_pydantic_core_schema__` 方法，用于在 Pydantic 模型中进行自定义验证
- 支持严格的类型检查，可验证值是否为 NumPy 数组以及正确的数据类型
- 支持上下文验证和配置选项

### Tensor 类型
- 在 `dataschemas/torch.py` 中定义
- 自定义的 `Tensor` 类型，继承自 `torch.Tensor`
- 支持条件导入 torch（如果未安装则抛出 ImportError）
- 提供了 `__get_pydantic_core_schema__` 方法，用于在 Pydantic 模型中进行自定义验证
- 支持将输入转换为 PyTorch 张量

### 基础元数据
- `MyBaseMetadata`: 抽象基类，提供元数据验证的基础功能 (`metadatas/basemetadata.py`)
- `SubscriptableMetadata`: 支持下标的元数据基类

### 设备元数据
- `Device` 元数据类，用于指定张量的设备（CPU/CUDA）(`metadatas/device.py`)
- `CPU` 和 `CUDA` 预定义设备实例
- `CUDA_` 生成器用于创建多个 CUDA 设备实例
- 支持张量的设备转换验证
- 根据 PyTorch 可用性动态初始化 CUDA 设备

### 梯度元数据
- `RequireGrad` 元数据类，用于控制张量是否需要梯度 (`metadatas/grad.py`)
- `WITH_GRAD` 和 `NO_GRAD` 预定义梯度设置实例
- 支持按位取反操作来切换梯度状态
- 只对浮点张量进行梯度设置，对非浮点张量会先转换为浮点

### 数字范围元数据
- `nrange` 元数据，用于数字范围验证 (`metadatas/nrange.py`)
- 支持使用 SymPy 符号表达式定义范围边界
- 提供类似 Python 切片语法的范围定义（如 `nrange[X:Y]`）
- 支持开区间、闭区间和半开区间验证
- 与上下文中的 SymPy 命名空间集成，实现动态范围验证
- 可与张量形状验证结合使用，实现基于变量的尺寸约束

### 张量形状元数据
- `tensorshape` 元数据，支持使用 SymPy 符号进行形状验证 (`metadatas/tensorshape.py`)
- 支持椭圆（...）和符号表达式进行形状匹配
- 允许形状变量定义和表达式计算
- 支持切片语法进行维度范围验证
- 通过 SymPy 表达式实现动态形状验证

### 协议定义
- `TensorLike` 协议，定义张量类对象的接口 (`metadatas/protocols.py`)
- 定义了形状、数据类型、维度等属性和基本计算操作
- 通过猴子补丁添加了 Pydantic 验证功能
- `SupportsCompute` 协议，定义支持计算操作的接口

### 猴子补丁功能
- `_patch/` 目录包含了对 Pydantic 和 SymPy 的猴子补丁实现
- `pydantic.py` 文件实现了对 `ValidateCallWrapper` 的修改，支持在 `validate_call` 装饰器中传递上下文信息
- `sympy.py` 文件为 SymPy 的 `Symbol` 和 `Expr` 类添加了 Pydantic 验证支持
- 通过添加 `context={}` 参数来启用上下文验证功能
- 支持在验证过程中共享 SymPy 符号命名空间，实现跨参数的符号约束

### SymPy 表达式验证
- 为 SymPy 的 `Symbol` 和 `Expr` 类添加了 Pydantic 验证支持 (`_patch/sympy.py`)
- 支持符号值的自动存储和一致性检查，确保同一符号在多次使用中保持相同值
- 支持复杂表达式的验证，通过符号替换验证表达式值与输入值的匹配
- 提供数值容差机制（默认 1e-10），处理浮点数精度问题
- 与上下文验证系统集成，实现跨参数的符号约束和表达式验证
- 支持在函数参数中使用 SymPy 符号作为类型注解，实现符号级别的类型安全

## 运行命令
- 运行测试: `uv run pytest`
- 类型检查: `uv run mypy .`
- 项目主要作为库使用，无直接运行入口（已移除 main.py）

## 开发实践
- 使用 Pydantic 进行数据验证
- 使用 NumPy 和 PyTorch 进行数值计算（可选依赖）
- 使用泛型类型确保类型安全
- 遵循 Python 3.12+ 的语法和特性
- 使用 uv 作为包管理器
- 使用可选依赖策略，核心功能不依赖 numpy 和 torch
- 使用自定义架构和元数据进行高级验证
- 支持 CUDA 设备的张量验证
- 利用 SymPy 符号进行动态形状验证
- 通过猴子补丁增强 Pydantic 功能