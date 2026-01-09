"""
Ellipsoid search space implementation.

This module implements the Ell class representing an ellipsoid search space
in the Ellipsoid Method.

Ell = {x | (x - xc)^T mq^-1 (x - xc) ≤ κ}
"""

from typing import Protocol, Tuple, Union, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .cutting_plane import CutStatus
from .ell_calc import EllCalc


@runtime_checkable
class UpdateByCutChoice(Protocol):
    """Protocol for types that can update cut choices."""

    def update_bias_cut_by(self, ell: "Ell", grad: NDArray[np.float64]) -> CutStatus:
        """Update using bias cut strategy."""
        ...

    def update_central_cut_by(self, ell: "Ell", grad: NDArray[np.float64]) -> CutStatus:
        """Update using central cut strategy."""
        ...

    def update_q_by(self, ell: "Ell", grad: NDArray[np.float64]) -> CutStatus:
        """Update using Q cut strategy."""
        ...


class Ell:
    """
    Ellipsoid search space in the Ellipsoid Method.

    Ell = {x | (x - xc)^T mq^-1 (x - xc) ≤ κ}

    Attributes:
        no_defer_trick: Whether to use the defer trick (improves efficiency)
        mq: Shape matrix of the ellipsoid (2D array)
        xc: Center of the ellipsoid (1D array)
        kappa: Size parameter of the ellipsoid
        ndim: Number of dimensions
        helper: EllCalc instance for calculations
        tsq: Squared Mahalanobis distance threshold
    """

    def __init__(
        self,
        kappa: float,
        mq: NDArray[np.float64],
        xc: NDArray[np.float64],
        no_defer_trick: bool = False,
    ) -> None:
        """
        Initialize an ellipsoid with given parameters.

        Args:
            kappa: Size parameter of the ellipsoid
            mq: Shape matrix (must be square and positive definite)
            xc: Center coordinates
            no_defer_trick: Whether to disable the defer trick
        """
        if mq.ndim != 2:
            raise ValueError("mq must be a 2D array")
        if xc.ndim != 1:
            raise ValueError("xc must be a 1D array")
        if mq.shape[0] != mq.shape[1]:
            raise ValueError("mq must be a square matrix")
        if mq.shape[0] != xc.shape[0]:
            raise ValueError("mq and xc must have compatible dimensions")

        self.kappa = kappa
        self.mq = mq.copy()
        self.xc = xc.copy()
        self.no_defer_trick = no_defer_trick
        self.helper = EllCalc(xc.shape[0])
        self.tsq = 0.0

    @classmethod
    def new_with_matrix(
        cls,
        kappa: float,
        mq: NDArray[np.float64],
        xc: NDArray[np.float64],
    ) -> "Ell":
        """
        Create a new ellipsoid with given matrix.

        Args:
            kappa: Size parameter
            mq: Shape matrix
            xc: Center coordinates

        Returns:
            New Ell instance
        """
        return cls(kappa, mq, xc)

    @classmethod
    def new(
        cls,
        val: NDArray[np.float64],
        xc: NDArray[np.float64],
    ) -> "Ell":
        """
        Create a new ellipsoid with diagonal matrix.

        Args:
            val: Diagonal values
            xc: Center coordinates

        Returns:
            New Ell instance
        """
        mq = np.diag(val)
        return cls.new_with_matrix(1.0, mq, xc)

    @classmethod
    def new_with_scalar(
        cls,
        val: float,
        xc: NDArray[np.float64],
    ) -> "Ell":
        """
        Create a new ellipsoid with scalar diagonal.

        Args:
            val: Scalar value for diagonal
            xc: Center coordinates

        Returns:
            New Ell instance
        """
        n = xc.shape[0]
        mq = val * np.eye(n)
        return cls.new_with_matrix(val, mq, xc)

    def _update_core(
        self,
        grad: NDArray[np.float64],
        beta: Union[float, Tuple[float, float]],
        cut_strategy,
    ) -> CutStatus:
        """
        Update ellipsoid core function using the cut.

        grad^T * (x - xc) + beta ≤ 0

        Args:
            grad: Gradient vector
            beta: Cut parameter(s)
            cut_strategy: Function to compute cut parameters

        Returns:
            Cut status
        """
        grad_t = self.mq @ grad
        omega = grad @ grad_t

        self.tsq = self.kappa * omega
        status, (rho, sigma, delta) = cut_strategy(beta, self.tsq)

        if status != CutStatus.SUCCESS:
            return status

        # Update center
        self.xc -= (rho / omega) * grad_t

        # Update shape matrix
        r = sigma / omega
        grad_t_2d = grad_t[:, np.newaxis]
        self.mq -= r * (grad_t_2d @ grad_t_2d.T)

        # Update size parameter
        self.kappa *= delta

        # Apply defer trick if disabled
        if self.no_defer_trick:
            self.mq *= self.kappa
            self.kappa = 1.0

        return status

    def update_bias_cut(
        self,
        cut: Tuple[NDArray[np.float64], Union[float, Tuple[float, float]]],
    ) -> CutStatus:
        """
        Update using bias cut strategy.

        Args:
            cut: Tuple of (gradient, beta)

        Returns:
            Cut status
        """
        grad, beta = cut

        if isinstance(beta, (int, float)):
            return self._update_core(
                grad,
                float(beta),
                lambda b, tsq: self.helper.calc_bias_cut(b, tsq),
            )
        else:
            # beta is (beta, parallel_beta)
            return self._update_core(
                grad,
                beta,
                lambda b, tsq: self.helper.calc_single_or_parallel_bias_cut(b, tsq),
            )

    def update_central_cut(
        self,
        cut: Tuple[NDArray[np.float64], Union[float, Tuple[float, float]]],
    ) -> CutStatus:
        """
        Update using central cut strategy.

        Args:
            cut: Tuple of (gradient, beta)

        Returns:
            Cut status
        """
        grad, beta = cut

        if isinstance(beta, (int, float)):
            return self._update_core(
                grad,
                float(beta),
                lambda b, tsq: self.helper.calc_central_cut(tsq),
            )
        else:
            # beta is (beta, parallel_beta)
            return self._update_core(
                grad,
                beta,
                lambda b, tsq: self.helper.calc_single_or_parallel_central_cut(b, tsq),
            )

    def update_q(
        self,
        cut: Tuple[NDArray[np.float64], Union[float, Tuple[float, float]]],
    ) -> CutStatus:
        """
        Update using Q cut strategy.

        Args:
            cut: Tuple of (gradient, beta)

        Returns:
            Cut status
        """
        grad, beta = cut

        if isinstance(beta, (int, float)):
            return self._update_core(
                grad,
                float(beta),
                lambda b, tsq: self.helper.calc_bias_cut_q(b, tsq),
            )
        else:
            # beta is (beta, parallel_beta)
            return self._update_core(
                grad,
                beta,
                lambda b, tsq: self.helper.calc_single_or_parallel_q(b, tsq),
            )

    def get_xc(self) -> NDArray[np.float64]:
        """Get the center coordinates."""
        return self.xc.copy()

    def get_tsq(self) -> float:
        """Get the squared Mahalanobis distance threshold."""
        return self.tsq

    def set_xc(self, x: NDArray[np.float64]) -> None:
        """Set the center coordinates."""
        if x.shape != self.xc.shape:
            raise ValueError(f"Expected shape {self.xc.shape}, got {x.shape}")
        self.xc = x.copy()

    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return self.xc.shape[0]

    def __repr__(self) -> str:
        return (
            f"Ell(kappa={self.kappa:.4f}, "
            f"ndim={self.ndim}, "
            f"xc={self.xc[:3]}{'...' if self.ndim > 3 else ''})"
        )
