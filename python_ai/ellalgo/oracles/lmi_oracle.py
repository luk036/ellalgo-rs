"""
Linear Matrix Inequality (LMI) Oracle.

This oracle handles Linear Matrix Inequality constraints of the form:
    F(x) = F₀ + Σ xᵢFᵢ ≽ 0
where Fᵢ are symmetric matrices and ≽ means positive semidefinite.
"""

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..cutting_plane import OracleFeas


class LmiOracle(OracleFeas):
    """
    Oracle for Linear Matrix Inequality constraints.

    Handles constraints of the form: F(x) = F₀ + Σ xᵢFᵢ ≽ 0
    """

    def __init__(
        self,
        F0: NDArray[np.float64],
        F_matrices: List[NDArray[np.float64]],
    ) -> None:
        """
        Initialize LMI oracle.

        Args:
            F0: Constant term matrix F₀
            F_matrices: List of coefficient matrices Fᵢ
        """
        self.F0 = F0.copy()
        self.F_matrices = [F.copy() for F in F_matrices]

        # Validate dimensions
        n = F0.shape[0]
        if F0.shape[1] != n:
            raise ValueError("F0 must be square")

        for F in self.F_matrices:
            if F.shape != (n, n):
                raise ValueError(f"All F matrices must be {n}x{n}")

    def assess_feas(
        self,
        xc: NDArray[np.float64],
    ) -> Optional[Tuple[NDArray[np.float64], float]]:
        """
        Assess feasibility at point xc.

        Args:
            xc: Current point

        Returns:
            None if feasible, otherwise (gradient, beta) cut information
        """
        # Compute F(x) = F₀ + Σ xᵢFᵢ
        self.F0.shape[0]
        F_x = self.F0.copy()

        for i, F_i in enumerate(self.F_matrices):
            if i < len(xc):
                F_x += xc[i] * F_i

        # Compute eigenvalues to check positive semidefiniteness
        eigenvalues = np.linalg.eigvalsh(F_x)
        min_eigenvalue = np.min(eigenvalues)

        if min_eigenvalue >= 0.0:
            # Feasible
            return None

        # Compute gradient using eigenvector corresponding to minimum eigenvalue
        eigvals, eigvecs = np.linalg.eigh(F_x)
        min_idx = np.argmin(eigvals)
        v = eigvecs[:, min_idx]

        # Gradient components: gᵢ = vᵀFᵢv
        grad = np.zeros(len(xc))
        for i, F_i in enumerate(self.F_matrices):
            if i < len(grad):
                grad[i] = v.T @ F_i @ v

        # Beta = vᵀF(x)v = min eigenvalue (negative)
        beta = min_eigenvalue

        return grad, beta

    def __repr__(self) -> str:
        n = self.F0.shape[0]
        m = len(self.F_matrices)
        return f"LmiOracle(n={n}, m={m})"
