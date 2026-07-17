"""Diagnostics mixin for BasanosEngine.

Provides matrix-quality and solver-quality properties as a reusable mixin so
that ``optimizer.py`` stays focused on the core position-solving logic.

Classes in this module are **private implementation details**.  The public API
is `BasanosEngine`, which inherits from
`_DiagnosticsMixin`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from cvx.linalg import SingularMatrixError, solve, valid

if TYPE_CHECKING:
    from ._engine_protocol import _EngineProtocol

_logger = logging.getLogger(__name__)


class _DiagnosticsMixin:
    """Mixin providing matrix-quality and solver-quality diagnostic properties.

    The consuming class must satisfy `_EngineProtocol`,
    i.e. it must expose:

    * ``assets`` — list of asset column names
    * ``prices`` — Polars DataFrame with a ``'date'`` column
    * ``mu`` — Polars DataFrame of expected-return signals
    * ``_iter_matrices()`` — generator yielding ``(i, t, mask, bundle)``
    """

    @property
    def condition_number(self: _EngineProtocol) -> pl.DataFrame:
        """Condition number κ of the effective correlation matrix at each timestamp.

        Uses the same covariance mode as `cash_position`: for
        ``ewma_shrink`` this is the shrunk EWMA matrix; for ``sliding_window``
        it is the factor-model covariance.  Only the sub-matrix corresponding
        to assets with finite prices at that timestamp is used; rows with no
        finite prices yield ``NaN``.

        Returns:
            pl.DataFrame: Two-column DataFrame ``{'date': ..., 'condition_number': ...}``.
        """
        kappas: list[float] = []
        for _i, _t, _mask, bundle in self._iter_matrices():
            if bundle is None:
                kappas.append(float(np.nan))
                continue
            _v, mat = valid(bundle.matrix)
            if not _v.any():
                kappas.append(float(np.nan))
                continue
            kappas.append(float(np.linalg.cond(mat)))

        return pl.DataFrame({"date": self.prices["date"], "condition_number": pl.Series(kappas, dtype=pl.Float64)})

    @property
    def effective_rank(self: _EngineProtocol) -> pl.DataFrame:
        r"""Effective rank of the effective correlation matrix at each timestamp.

        Measures the true dimensionality of the portfolio by computing the
        entropy-based effective rank:

        $$
        \\text{eff\\_rank} = \\exp\\!\\left(-\\sum_i p_i \\ln p_i\\right),
        \\quad p_i = \\frac{\\lambda_i}{\\sum_j \\lambda_j}
        $$

        where $\\lambda_i$ are the eigenvalues of the effective
        correlation matrix (restricted to assets with finite prices at that
        timestamp).  Uses the same covariance mode as `cash_position`.
        A value equal to the number of assets indicates a perfectly uniform
        spectrum; a value of 1 indicates a rank-1 matrix.

        Returns:
            pl.DataFrame: Two-column DataFrame ``{'date': ..., 'effective_rank': ...}``.
        """
        ranks: list[float] = []
        for _i, _t, _mask, bundle in self._iter_matrices():
            if bundle is None:
                ranks.append(float(np.nan))
                continue
            _v, mat = valid(bundle.matrix)
            if not _v.any():
                ranks.append(float(np.nan))
                continue
            eigvals = np.linalg.eigvalsh(mat)
            eigvals = np.clip(eigvals, 0.0, None)
            total = eigvals.sum()
            if total <= 0.0:
                ranks.append(float(np.nan))
                continue
            p = eigvals / total
            p_pos = p[p > 0.0]
            entropy = float(-np.sum(p_pos * np.log(p_pos)))
            ranks.append(float(np.exp(entropy)))

        return pl.DataFrame({"date": self.prices["date"], "effective_rank": pl.Series(ranks, dtype=pl.Float64)})

    @staticmethod
    def _residual_for_row(matrix: np.ndarray, expected_mu: np.ndarray, t: object) -> float:
        """Return the solver residual for a single timestamp.

        Returns ``0.0`` for an all-zero signal (no solve performed) and
        ``NaN`` when the matrix is singular or the solution has no finite
        entries.
        """
        if np.allclose(expected_mu, 0.0):
            return 0.0
        try:
            x = solve(matrix, expected_mu)
        except SingularMatrixError:
            # The covariance matrix is degenerate — residual is undefined.
            _logger.warning(
                "solver_residual: SingularMatrixError at t=%s - covariance matrix is degenerate; residual set to NaN.",
                t,
            )
            return float(np.nan)
        finite_x = np.isfinite(x)
        if not finite_x.any():
            return float(np.nan)
        return float(np.linalg.norm(matrix[np.ix_(finite_x, finite_x)] @ x[finite_x] - expected_mu[finite_x]))

    @property
    def solver_residual(self: _EngineProtocol) -> pl.DataFrame:
        r"""Per-timestamp solver residual ``‖C·x - μ‖₂``.

        After solving the normalised linear system ``C · x = μ`` at
        each timestamp, this property reports the Euclidean residual norm.
        For a well-posed, well-conditioned system the residual is near machine
        epsilon; large values flag numerical difficulties (near-singular
        matrices, extreme condition numbers, or solver fall-back to LU).
        Uses the same covariance mode as `cash_position`.

        Returns:
            pl.DataFrame: Two-column DataFrame ``{'date': ..., 'residual': ...}``.
            Zero is returned when ``μ`` is the zero vector (no solve is
            performed).  ``NaN`` is returned when no asset has finite prices.
        """
        assets = self.assets
        mu_np = self.mu.select(assets).to_numpy()

        residuals: list[float] = []
        for i, t, mask, bundle in self._iter_matrices():
            if bundle is None:
                residuals.append(float(np.nan))
                continue
            expected_mu = np.nan_to_num(mu_np[i][mask])
            residuals.append(_DiagnosticsMixin._residual_for_row(bundle.matrix, expected_mu, t))

        return pl.DataFrame({"date": self.prices["date"], "residual": pl.Series(residuals, dtype=pl.Float64)})

    @staticmethod
    def _utilisation_for_row(
        matrix: np.ndarray, expected_mu: np.ndarray, t: object, mu_tol: float
    ) -> np.ndarray | None:
        """Return the per-asset utilisation ratios for a single timestamp.

        Returns an all-zero vector for an all-zero signal (no solve performed)
        and ``None`` when the matrix is singular (utilisation left as ``NaN``).
        """
        if np.allclose(expected_mu, 0.0):
            return np.zeros_like(expected_mu)
        try:
            x = solve(matrix, expected_mu)
        except SingularMatrixError:
            # The covariance matrix is degenerate — utilisation is undefined.
            _logger.warning(
                "signal_utilisation: SingularMatrixError at t=%s - covariance matrix is "
                "degenerate; utilisation set to NaN.",
                t,
            )
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(expected_mu) > mu_tol, x / expected_mu, np.nan)

    @property
    def signal_utilisation(self: _EngineProtocol) -> pl.DataFrame:
        r"""Per-asset signal utilisation: fraction of μ_i surviving the correlation filter.

        For each asset *i* and timestamp *t*, computes

        $$
        u_i = \\frac{(C^{-1}\\,\\mu)_i}{\\mu_i}
        $$

        where $C^{-1}\\,\\mu$ is the unnormalised solve result using
        the effective correlation matrix for the current
        `covariance_mode`.  When $C = I$
        (identity) all assets have utilisation 1.  Off-diagonal correlations
        attenuate some assets ($u_i < 1$) and may amplify negatively
        correlated ones ($u_i > 1$).

        A value of ``0.0`` is returned when the entire signal vector
        $\\mu$ is near zero at that timestamp (no solve is performed).
        ``NaN`` is returned for individual assets where $|\\mu_i|$ is
        below machine-epsilon precision or where prices are unavailable.

        Returns:
            pl.DataFrame: DataFrame with columns ``['date'] + assets``.
        """
        assets = self.assets
        mu_np = self.mu.select(assets).to_numpy()

        _mu_tol = 1e-14  # treat |μ_i| below this as zero to avoid spurious large ratios
        n_assets = len(assets)
        util_np = np.full((self.prices.height, n_assets), np.nan)

        for i, t, mask, bundle in self._iter_matrices():
            if bundle is None:
                continue
            expected_mu = np.nan_to_num(mu_np[i][mask])
            ratio = _DiagnosticsMixin._utilisation_for_row(bundle.matrix, expected_mu, t, _mu_tol)
            if ratio is not None:
                util_np[i, mask] = ratio

        return self.prices.with_columns([pl.lit(util_np[:, j]).alias(asset) for j, asset in enumerate(assets)])
