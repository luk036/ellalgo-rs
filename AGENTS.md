# AGENTS.md - Agent Coding Guidelines

## Build, Lint, and Test Commands

### Essential Commands
```bash
# Run all tests
cargo test --all-features --workspace

# Run a single test (e.g., test_profit_oracle)
cargo test test_profit_oracle

# Build and run release version
cargo build --release && cargo run --release

# Run Clippy linter
cargo clippy --all-targets --all-features --workspace

# Check code formatting
cargo fmt --all -- --check

# Format code
cargo fmt --all

# Build documentation
cargo doc --no-deps --document-private-items --all-features --workspace --examples
```

### Test Workflow
- Run full test suite before committing
- Use regression tests with fixed iteration counts
- Tests are organized in `#[cfg(test)]` modules within source files
- Integration tests in dedicated test modules

## Code Style Guidelines

### Import Organization
```rust
// First: crate-local imports from parent modules
use super::cutting_plane::OracleOptim;
use crate::ell::Ell;

// Second: external crate imports
use ndarray::prelude::*;
use approx_eq::assert_approx_eq;

// Type aliases typically follow imports
type Arr = Array1<f64>;
```

### Naming Conventions
- **Functions & Methods**: `snake_case`
- **Structs & Enums**: `PascalCase`
- **Traits**: `PascalCase` (e.g., `OracleFeas`, `SearchSpace`)
- **Constants**: `UPPER_SNAKE_CASE`
- **Variables**: `snake_case` (mathematical variables may use single letters)
- **Associated Types**: `PascalCase` (e.g., `type CutChoice = f64;`)

### Type System
- Use type aliases for common array types: `type Arr = Array1<f64>;`
- Leverage associated types in traits for flexibility
- Use generics extensively for reusable code
- Prefer `Option<T>` for fallible operations over `Result<T, E>`

### Error Handling
```rust
// Prefer Option for fallible operations
fn assess_feas(&mut self, xc: &ArrayType) -> Option<(ArrayType, Self::CutChoice)> {
    if condition {
        return Some((gradient, beta));
    }
    None  // Feasible solution found
}

// Custom status enums for algorithms
pub enum CutStatus {
    Success,
    NoSoln,
    NoEffect,
    Unknown,
}
```

### Documentation Style
```rust
/// The function description with mathematical context.
///
/// Arguments:
///
/// * `param1`: Description of what it represents
/// * `param2`: Description with units or constraints
///
/// Returns:
///
/// Description of return value and its meaning.
///
/// # Examples
///
/// ```
/// use crate_name::struct_name::StructName;
/// let instance = StructName::new(params);
/// assert_eq!(instance.field, expected_value);
/// ```
```

### Testing Patterns
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cutting_plane::{cutting_plane_optim, Options};
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_descriptive_name() {
        // Arrange
        let mut oracle = MyOracle::new(params);
        let mut ellip = Ell::new(array![10.0, 10.0], array![0.0, 0.0]);
        let mut gamma = f64::NEG_INFINITY;
        let options = Options::default();

        // Act
        let (xbest, num_iters) = cutting_plane_optim(&mut oracle, &mut ellip, &mut gamma, &options);

        // Assert
        assert!(xbest.is_some());
        assert_eq!(num_iters, 25, "regression test");
    }
}
```

### Trait Design Patterns
```rust
pub trait OracleOptim<ArrayType> {
    type CutChoice;  // Associated type for flexibility (f64 or tuple)

    fn assess_optim(
        &mut self,
        xc: &ArrayType,
        gamma: &mut f64,
    ) -> ((ArrayType, Self::CutChoice), bool);
}
```

### Common Patterns

1. **Type Aliases**: Define at module level for common types
2. **Helper Structs**: Use for algorithm parameters (e.g., `EllCalcCore`, `Options`)
3. **Default Implementations**: Provide sensible defaults via `Default` trait
4. **Clone**: Implement `Clone` for small structs used in algorithms
5. **Math Documentation**: Include mathematical formulas and context for optimization structures

### Specific Conventions

- Use `ndarray` for all array operations
- Floating-point comparisons use `approx_eq::assert_approx_eq!`
- Round-robin constraint checking: maintain `idx` field and reset with modulo
- Oracle traits take mutable `&mut self` for statefulness
- Search space traits use `UpdateByCutChoice` for flexible update strategies
- Prefer `const fn` for constructors that can be evaluated at compile time
- Use `#![allow(non_snake_case)]` sparingly, only for mathematical variable names

### Module Organization
- `lib.rs`: Public API and module declarations
- `cutting_plane.rs`: Core algorithm traits and functions
- `ell*.rs`: Ellipsoid search space implementations
- `oracles/`: Problem-specific oracle implementations
- `example*.rs`: Example oracles for testing

### Performance Notes
- Use `Array2::eye()` and `Array2::from_diag()` for matrix construction
- Prefer `&Array1<f64>` over `Array1<f64>` for read-only references
- Use `mapv()` for element-wise operations with closures
- Defer matrix updates when possible (see `no_defer_trick` pattern)

## CI Requirements

All changes must pass:
1. `cargo fmt --all -- --check` - Formatting
2. `cargo clippy --all-targets` - Linting
3. `cargo test --all-features --workspace` - Tests
4. `cargo doc` with `-D warnings` - Documentation
