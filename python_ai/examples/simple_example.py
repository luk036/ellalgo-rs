"""
Simple example of using the ellalgo library.
"""

import numpy as np

from ellalgo import CutStatus, Ell, cutting_plane


class SimpleOracle:
    """Simple feasibility oracle for unit ball constraint."""

    def assess_feas(self, xc: np.ndarray):
        """Check if point is inside unit ball."""
        norm_sq = np.sum(xc**2)
        if norm_sq <= 1.0:
            return None  # Feasible

        # Return cut: gradient = 2xc, beta = 1 - ||xc||²
        grad = 2.0 * xc
        beta = 1.0 - norm_sq
        return grad, beta


def main():
    """Run simple example."""
    print("Simple Ellipsoid Method Example")
    print("=" * 40)

    # Create ellipsoid centered at origin with radius 2
    xc = np.array([2.0, 2.0, 2.0])  # Start outside unit ball
    ell = Ell.new_with_scalar(4.0, xc)  # Initial ellipsoid radius = 2

    # Create oracle
    oracle = SimpleOracle()

    # Run cutting plane algorithm
    status, solution, _, iterations = cutting_plane(oracle, ell)

    print(f"Status: {status}")
    print(f"Solution: {solution}")
    print(f"Iterations: {iterations}")
    print(f"Norm: {np.linalg.norm(solution):.6f}")

    if status == CutStatus.SUCCESS:
        print("✓ Found feasible solution inside unit ball")
    else:
        print("✗ Failed to find feasible solution")


if __name__ == "__main__":
    main()
