"""
Ellipsoid calculation utilities.

This module provides calculation functions for updating ellipsoid parameters
in the Ellipsoid Method.
"""

from typing import Optional, Tuple

import numpy as np

from .cutting_plane import CutStatus


class EllCalcCore:
    """
    Core parameters for ellipsoid calculations.

    Pre-computes constants that depend on the dimension n.

    Attributes:
        n_f: Dimension n as float
        n_plus_1: n + 1
        half_n: n / 2
        inv_n: 1 / n
        cst1: n² / (n² - 1)
        cst2: 2 / (n + 1)
    """

    def __init__(self, n_f: float) -> None:
        """
        Initialize with dimension n.

        Args:
            n_f: Dimension as float
        """
        self.n_f = n_f
        self.n_plus_1 = n_f + 1.0
        self.half_n = n_f / 2.0
        self.inv_n = 1.0 / n_f

        n_sq = n_f * n_f
        cst0 = 1.0 / (n_f + 1.0)
        self.cst1 = n_sq / (n_sq - 1.0)
        self.cst2 = 2.0 * cst0

    def calc_parallel_bias_cut_fast(
        self,
        beta0: float,
        beta1: float,
        tsq: float,
        eta: float,
        beta_sq: float,
    ) -> Tuple[float, float, float]:
        """
        Calculate parameters for parallel bias cut (fast version).

        Args:
            beta0: First beta parameter
            beta1: Second beta parameter
            tsq: τ² (squared semi-major axis)
            eta: η parameter
            beta_sq: β² = β0 * β1

        Returns:
            Tuple of (rho, sigma, delta)
        """
        beta = (beta0 + beta1) / 2.0
        h = 0.5 * (tsq + beta_sq) + self.n_f * beta * beta
        k = h + np.sqrt(h * h - self.n_plus_1 * eta * beta_sq)

        sigma = eta / k
        mu_inv = eta / (k - eta)
        rho = beta * sigma
        delta = (tsq + (beta_sq * sigma - beta0 * beta1) * mu_inv) / tsq

        return rho, sigma, delta

    def calc_parallel_bias_cut(
        self,
        beta0: float,
        beta1: float,
        tsq: float,
    ) -> Tuple[float, float, float]:
        """
        Calculate parameters for parallel bias cut.

        Args:
            beta0: First beta parameter
            beta1: Second beta parameter
            tsq: τ² (squared semi-major axis)

        Returns:
            Tuple of (rho, sigma, delta)
        """
        beta_sq = beta0 * beta1
        eta = self.cst1
        return self.calc_parallel_bias_cut_fast(beta0, beta1, tsq, eta, beta_sq)

    def calc_parallel_central_cut(
        self,
        beta1: float,
        tsq: float,
    ) -> Tuple[float, float, float]:
        """
        Calculate parameters for parallel central cut.

        Args:
            beta1: Beta parameter
            tsq: τ² (squared semi-major axis)

        Returns:
            Tuple of (rho, sigma, delta)
        """
        beta_sq = -beta1 * beta1
        eta = self.cst1
        return self.calc_parallel_bias_cut_fast(-beta1, beta1, tsq, eta, beta_sq)

    def calc_central_cut(self, tsq: float) -> Tuple[float, float, float]:
        """
        Calculate parameters for central cut.

        Args:
            tsq: τ² (squared semi-major axis)

        Returns:
            Tuple of (rho, sigma, delta)
        """
        rho = 0.0
        sigma = self.cst2
        delta = (self.n_f * self.n_f) / (self.n_plus_1 * self.n_plus_1)
        return rho, sigma, delta


class EllCalc:
    """
    Ellipsoid calculation helper.

    Provides methods for calculating cut parameters with status checking.
    """

    def __init__(self, n: int) -> None:
        """
        Initialize with dimension n.

        Args:
            n: Dimension (number of variables)
        """
        self.core = EllCalcCore(float(n))

    def calc_single_or_parallel_bias_cut(
        self,
        beta_pair: Tuple[float, Optional[float]],
        tsq: float,
    ) -> Tuple[CutStatus, Tuple[float, float, float]]:
        """
        Calculate bias cut parameters for single or parallel cut.

        Args:
            beta_pair: (beta, parallel_beta) where parallel_beta can be None
            tsq: τ² (squared semi-major axis)

        Returns:
            Tuple of (status, (rho, sigma, delta))
        """
        beta, parallel_beta = beta_pair

        if parallel_beta is None:
            return self.calc_bias_cut(beta, tsq)
        else:
            return self.calc_parallel_bias_cut(beta, parallel_beta, tsq)

    def calc_single_or_parallel_central_cut(
        self,
        beta_pair: Tuple[float, Optional[float]],
        tsq: float,
    ) -> Tuple[CutStatus, Tuple[float, float, float]]:
        """
        Calculate central cut parameters for single or parallel cut.

        Args:
            beta_pair: (beta, parallel_beta) where parallel_beta can be None
            tsq: τ² (squared semi-major axis)

        Returns:
            Tuple of (status, (rho, sigma, delta))
        """
        beta, parallel_beta = beta_pair

        if parallel_beta is None:
            return self.calc_central_cut(tsq)
        else:
            return self.calc_parallel_central_cut(parallel_beta, tsq)

    def calc_single_or_parallel_q(
        self,
        beta_pair: Tuple[float, Optional[float]],
        tsq: float,
    ) -> Tuple[CutStatus, Tuple[float, float, float]]:
        """
        Calculate Q cut parameters for single or parallel cut.

        Args:
            beta_pair: (beta, parallel_beta) where parallel_beta can be None
            tsq: τ² (squared semi-major axis)

        Returns:
            Tuple of (status, (rho, sigma, delta))
        """
        beta, parallel_beta = beta_pair

        if parallel_beta is None:
            return self.calc_bias_cut_q(beta, tsq)
        else:
            return self.calc_parallel_q(beta, parallel_beta, tsq)

    def calc_parallel_bias_cut(
        self,
        beta0: float,
        beta1: float,
        tsq: float,
    ) -> Tuple[CutStatus, Tuple[float, float, float]]:
        """
        Calculate parallel bias cut with status checking.

        Args:
            beta0: First beta parameter
            beta1: Second beta parameter
            tsq: τ² (squared semi-major axis)

        Returns:
            Tuple of (status, (rho, sigma, delta))
        """
        if tsq <= 0.0:
            return CutStatus.NO_EFFECT, (0.0, 0.0, 1.0)

        beta_sq = beta0 * beta1
        if beta_sq <= 0.0:
            return CutStatus.NO_SOLN, (0.0, 0.0, 1.0)

        rho, sigma, delta = self.core.calc_parallel_bias_cut(beta0, beta1, tsq)
        return CutStatus.SUCCESS, (rho, sigma, delta)

    def calc_parallel_q(
        self,
        beta0: float,
        beta1: float,
        tsq: float,
    ) -> Tuple[CutStatus, Tuple[float, float, float]]:
        """
        Calculate parallel Q cut.

        Args:
            beta0: First beta parameter
            beta1: Second beta parameter
            tsq: τ² (squared semi-major axis)

        Returns:
            Tuple of (status, (rho, sigma, delta))
        """
        if tsq <= 0.0:
            return CutStatus.NO_EFFECT, (0.0, 0.0, 1.0)

        beta_sq = beta0 * beta1
        if beta_sq <= 0.0:
            return CutStatus.NO_SOLN, (0.0, 0.0, 1.0)

        # For Q cut, we use different constants
        eta = 1.0 / (self.core.n_f + 1.0)
        rho, sigma, delta = self.core.calc_parallel_bias_cut_fast(
            beta0, beta1, tsq, eta, beta_sq
        )
        return CutStatus.SUCCESS, (rho, sigma, delta)

    def calc_parallel_central_cut(
        self,
        beta1: float,
        tsq: float,
    ) -> Tuple[CutStatus, Tuple[float, float, float]]:
        """
        Calculate parallel central cut with status checking.

        Args:
            beta1: Beta parameter
            tsq: τ² (squared semi-major axis)

        Returns:
            Tuple of (status, (rho, sigma, delta))
        """
        if tsq <= 0.0:
            return CutStatus.NO_EFFECT, (0.0, 0.0, 1.0)

        rho, sigma, delta = self.core.calc_parallel_central_cut(beta1, tsq)
        return CutStatus.SUCCESS, (rho, sigma, delta)

    def calc_bias_cut(
        self,
        beta: float,
        tsq: float,
    ) -> Tuple[CutStatus, Tuple[float, float, float]]:
        """
        Calculate bias cut with status checking.

        Args:
            beta: Beta parameter
            tsq: τ² (squared semi-major axis)

        Returns:
            Tuple of (status, (rho, sigma, delta))
        """
        if tsq <= 0.0:
            return CutStatus.NO_EFFECT, (0.0, 0.0, 1.0)

        if beta >= 0.0:
            return CutStatus.NO_SOLN, (0.0, 0.0, 1.0)

        tau = np.sqrt(tsq)
        if beta <= -tau:
            return CutStatus.NO_EFFECT, (0.0, 0.0, 1.0)

        n = self.core.n_f

        rho = (n + 1.0) * beta / (n + 1.0)
        sigma = 2.0 * (n + 1.0) * (tau + beta) / ((n + 2.0) * tau + (n * beta))
        delta = (n * n * (tsq - beta * beta)) / ((n + 1.0) * (n + 1.0) * tsq)

        return CutStatus.SUCCESS, (rho, sigma, delta)

    def calc_bias_cut_q(
        self,
        beta: float,
        tsq: float,
    ) -> Tuple[CutStatus, Tuple[float, float, float]]:
        """
        Calculate bias Q cut with status checking.

        Args:
            beta: Beta parameter
            tsq: τ² (squared semi-major axis)

        Returns:
            Tuple of (status, (rho, sigma, delta))
        """
        if tsq <= 0.0:
            return CutStatus.NO_EFFECT, (0.0, 0.0, 1.0)

        if beta >= 0.0:
            return CutStatus.NO_SOLN, (0.0, 0.0, 1.0)

        tau = np.sqrt(tsq)
        if beta <= -tau:
            return CutStatus.NO_EFFECT, (0.0, 0.0, 1.0)

        n = self.core.n_f

        # Different constants for Q cut
        rho = beta / (n + 1.0)
        sigma = 2.0 * (tau + beta) / ((n + 2.0) * tau + n * beta)
        delta = (n * n * (tsq - beta * beta)) / ((n + 1.0) * (n + 1.0) * tsq)

        return CutStatus.SUCCESS, (rho, sigma, delta)

    def calc_central_cut(
        self,
        tsq: float,
    ) -> Tuple[CutStatus, Tuple[float, float, float]]:
        """
        Calculate central cut with status checking.

        Args:
            tsq: τ² (squared semi-major axis)

        Returns:
            Tuple of (status, (rho, sigma, delta))
        """
        if tsq <= 0.0:
            return CutStatus.NO_EFFECT, (0.0, 0.0, 1.0)

        rho, sigma, delta = self.core.calc_central_cut(tsq)
        return CutStatus.SUCCESS, (rho, sigma, delta)
