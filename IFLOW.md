# IFLOW.md - ellalgo-rs

## 项目概述

`ellalgo-rs` 是一个用 Rust 实现的椭球体方法（Ellipsoid Method）库。椭球体方法是由 L. G. Khachiyan 在 1979 年首次提出的线性规划算法，它是一个多项式时间算法，通过使用椭球体迭代缩小线性程序的可行区域，直到找到最优解。该算法保证在有限步数内收敛到最优解。

此 Rust 库实现了椭球体方法，并支持并行切面（Parallel Cut）以提高收敛速度，特别是在具有多个平行约束的问题中。该项目广泛使用了 Rust 的 ndarray 库进行数值计算。

## 核心组件

1. `Ell` 结构体 (`src/ell.rs`): 表示椭球体搜索空间，包含椭球体的形状矩阵、中心点、大小参数等，并实现了切面更新的核心逻辑。

2. `EllCalc` 结构体 (`src/ell_calc.rs`): 负责椭球体更新的计算，包括偏置切面、中心切面和平行切面的计算。

3. `cutting_plane.rs`: 定义了切面算法的核心 trait 和函数，包括 `Oracle` trait（用于评估可行性和优化）和 `SearchSpace` trait（表示搜索空间）。

4. `oracles` 模块 (`src/oracles/`): 包含各种特定问题的 Oracle 实现，用于提供切面信息。

## 构建和运行

### 依赖项
- Rust 工具链 (Cargo)
- `ndarray` 库 (v0.16.1)
- `svgbobdoc` 库 (v0.3)
- `approx_eq` 库 (v0.1.8, 用于测试)

### 构建命令
```bash
# 构建库
cargo build

# 构建并运行测试
cargo test

# 构建发布版本
cargo build --release
```

### 安装
```bash
# 全局安装
cargo install ellalgo-rs
```

## 开发约定

1. 代码风格：遵循 Rust 标准代码风格，使用 `rustfmt` 进行格式化。
2. 测试：核心功能都有相应的单元测试，使用 `cargo test` 运行。
3. 文档：使用 Rustdoc 进行代码文档化，包含示例代码。
4. 数值计算：广泛使用 `ndarray` 库进行数组和矩阵操作。

## 项目结构

- `src/`: 源代码目录
  - `ell.rs`: 椭球体实现
  - `ell_calc.rs`: 椭球体计算逻辑
  - `cutting_plane.rs`: 切面算法核心 trait 和函数
  - `oracles/`: 各种 Oracle 实现
  - `example*.rs`: 示例代码
  - `lib.rs`: 库的入口点

## 许可证

此项目根据 MIT 许可证或 Apache 许可证 2.0 版（您可选择）进行许可。