# ellalgo-py

Python implementation of the Ellipsoid Method with type annotations.

This is a Python port of the Rust `ellalgo-rs` library, implementing the Ellipsoid Method
(L. G. Khachiyan, 1979) for linear programming and convex optimization.

## Features

- **Ellipsoid Method**: Polynomial-time algorithm for linear programming
- **Parallel Cut Support**: Improved convergence with multiple parallel constraints
- **Type Annotations**: Full type hints for better IDE support and static analysis
- **NumPy Integration**: Efficient numerical computations using NumPy arrays
- **Comprehensive Testing**: Pytest test suite with property-based testing

## Installation

```bash
# Install from source
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Usage

```python
import numpy as np
from ellalgo import Ell, cutting_plane

# Create an ellipsoid search space
ell = Ell(kappa=1.0, mq=np.eye(3), xc=np.zeros(3))

# Define an oracle (feasibility/optimization evaluator)
class MyOracle:
    def assess_feasibility(self, x: np.ndarray):
        # Return cut information
        pass

# Run the cutting plane algorithm
result = cutting_plane(ell, MyOracle(), max_iters=1000, tolerance=1e-6)
```

## Project Structure

```
ellalgo-py/
├── ellalgo/
│   ├── __init__.py
│   ├── ell.py              # Ellipsoid implementation
│   ├── ell_calc.py         # Ellipsoid calculations
│   ├── cutting_plane.py    # Core algorithms
│   └── oracles/            # Oracle implementations
│       ├── __init__.py
│       ├── lmi_oracle.py   # Linear Matrix Inequality oracle
│       └── profit_oracle.py
├── tests/
│   ├── test_ell.py
│   ├── test_ell_calc.py
│   └── test_cutting_plane.py
├── pyproject.toml
└── README.md
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=ellalgo

# Type checking
mypy ellalgo/

# Format code
black ellalgo/ tests/
ruff check --fix ellalgo/ tests/
```

## License

MIT OR Apache-2.0 (same as the original Rust project)