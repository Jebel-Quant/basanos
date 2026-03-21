"""Solve/position mixin for BasanosEngine.

This private module contains :class:`_SolveMixin`, which provides the
``_iter_matrices`` and ``_iter_solve`` generator methods.  Separating them
from :mod:`basanos.math.optimizer` keeps the engine facade lean and makes
the per-timestamp solve logic independently readable and testable.
"""

from __future__ import annotations

import dataclasses
import logging
from enum import StrEnum
from typing import TYPE_CHECKING, cast

import numpy as np

from ..exceptions import SingularMatrixError
from ._config import EwmaShrinkConfig, SlidingWindowConfig
from ._ewm_corr import _ewm_corr_with_final_state, _EwmCorrState
from ._factor_model import FactorModel
from ._linalg import inv_a_norm, solve
from ._signal import shrink2id

if TYPE_CHECKING:
    from ._engine_protocol import _EngineProtocol

_logger = logging.getLogger(__name__)


class SolveStatus(StrEnum):
    """Solver outcome labels for each timestamp.

    Since :class:`SolveStatus` inherits from :class:`str` via ``StrEnum``,
    values compare equal to their string equivalents (e.g.
    ``SolveStatus.VALID == "valid"``), preserving backward compatibility
    with code that matches on string literals.

    Attributes:
        WARMUP: Insufficient history for the sliding-window covariance mode.
        ZERO_SIGNAL: The expected-return vector was all-zero; positions zeroed.
        DEGENERATE: Normalisation denominator was non-finite, solve failed, or
            no asset had a finite price; positions zeroed for safety.
        VALID: Linear system solved successfully; positions are non-trivially
            non-zero.
    """

    WARMUP = "warmup"
    ZERO_SIGNAL = "zero_signal"
    DEGENERATE = "degenerate"
    VALID = "valid"


@dataclasses.dataclass(frozen=True)
class WarmupState:
    """Final state produced by a full batch solve; consumed by :meth:`BasanosStream.from_warmup`.

    Returned by :meth:`BasanosEngine.warmup_state` and used by
    :meth:`BasanosStream.from_warmup` to initialise the streaming state without
    coupling to the private :meth:`~_SolveMixin._iter_solve` generator.

    Attributes:
        profit_variance: Final EWMA profit-variance scalar after replaying the
            full warmup batch.
        prev_cash_pos: Cash positions at the last warmup row, shape
            ``(n_assets,)``.  ``NaN`` for assets that were still in their
            own warmup period.
        corr_iir_state: Final IIR filter memory from the EWM correlation pass,
            or ``None`` when using :class:`~basanos.math.SlidingWindowConfig`.
            :meth:`BasanosStream.from_warmup` reads these arrays to seed the
            incremental ``lfilter`` state without a second pass over the
            warmup data.
    """

    profit_variance: float
    prev_cash_pos: np.ndarray
    corr_iir_state: _EwmCorrState | None = dataclasses.field(default=None)


class _SolveMixin:
    """Mixin that provides ``_iter_matrices`` and ``_iter_solve`` generators.

    Consumers must also inherit from (or satisfy the interface of)
    :class:`~basanos.math._engine_protocol._EngineProtocol` so that
    ``self.assets``, ``self.prices``, ``self.mu``, ``self.cfg``, ``self.cor``,
    and ``self.ret_adj`` are all available.
    """

    @staticmethod
    def _compute_mask(prices_row: np.ndarray) -> np.ndarray:
        """Return boolean mask indicating which assets have finite prices in the given row."""
        return np.isfinite(prices_row)

    @staticmethod
    def _check_signal(mu: np.ndarray, mask: np.ndarray) -> SolveStatus | None:
        """Return ``ZERO_SIGNAL`` when the masked expected-return vector is all-zero.

        Returns ``None`` when the signal is non-trivially non-zero, indicating
        that the caller should proceed to the linear solve.
        """
        if np.allclose(np.nan_to_num(mu[mask]), 0.0):
            return SolveStatus.ZERO_SIGNAL
        return None

    @staticmethod
    def _scale_to_cash(pos: np.ndarray, profit_variance: float, vola_active: np.ndarray) -> np.ndarray:
        """Convert raw solver positions to cash-adjusted positions.

        Divides *pos* by *profit_variance* to get risk positions, then divides
        by *vola_active* (volatility for the active asset subset) to get cash
        positions.  ``np.errstate(invalid="ignore")`` is applied internally so
        NaN volatility values propagate quietly.
        """
        risk_pos = pos / profit_variance
        with np.errstate(invalid="ignore"):
            return risk_pos / vola_active

    @staticmethod
    def _prepare_mu(mu_row: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, SolveStatus | None]:
        """Return ``(expected_mu, sig_status)`` for the active-asset subset.

        Combines :meth:`_check_signal` and ``nan_to_num`` extraction so both
        branches of :meth:`_iter_solve` avoid repeating the same two lines.

        Returns:
            tuple: ``(expected_mu, sig_status)`` — NaN-filled expected returns
            restricted to active assets, and :attr:`SolveStatus.ZERO_SIGNAL`
            when all active returns are zero (``None`` otherwise).
        """
        sig_status = _SolveMixin._check_signal(mu_row, mask)
        return np.nan_to_num(mu_row[mask]), sig_status

    @staticmethod
    def _solve_ewma_step(
        expected_mu: np.ndarray,
        matrix: np.ndarray,
        t: object,
        denom_tol: float,
    ) -> tuple[np.ndarray, SolveStatus]:
        """Run the EWMA linear solve for one timestamp.

        Computes the normalisation denominator via
        :func:`~basanos.math._linalg.inv_a_norm`, validates it against
        *denom_tol*, then solves via :func:`~basanos.math._linalg.solve`.

        Returns:
            tuple: ``(pos, status)`` — raw position vector before
            ``profit_variance`` scaling.  All failure paths return a zero
            vector with :attr:`SolveStatus.DEGENERATE`.
        """
        try:
            denom = inv_a_norm(expected_mu, matrix)
        except SingularMatrixError:
            denom = float("nan")
        if not np.isfinite(denom) or denom <= denom_tol:
            _logger.warning(
                "Positions zeroed at t=%s: normalisation denominator is degenerate "
                "(denom=%s, denom_tol=%s). Check signal magnitude and covariance matrix.",
                t,
                denom,
                denom_tol,
                extra={"context": {"t": str(t), "denom": denom, "denom_tol": denom_tol}},
            )
            return np.zeros_like(expected_mu), SolveStatus.DEGENERATE
        try:
            pos = solve(matrix, expected_mu) / denom
        except SingularMatrixError:
            _logger.warning("EWMA linear solve failed at t=%s: singular matrix.", t)
            return np.zeros_like(expected_mu), SolveStatus.DEGENERATE
        return pos, SolveStatus.VALID

    @staticmethod
    def _solve_sw_step(
        expected_mu: np.ndarray,
        fm: FactorModel,
        t: object,
        denom_tol: float,
    ) -> tuple[np.ndarray, SolveStatus]:
        """Run the sliding-window Woodbury solve for one timestamp.

        Solves via :meth:`~basanos.math._factor_model.FactorModel.solve`,
        derives the normalisation denominator, validates it, and normalises.

        Returns:
            tuple: ``(pos, status)`` — raw position vector before
            ``profit_variance`` scaling.  All failure paths return a zero
            vector with :attr:`SolveStatus.DEGENERATE`.
        """
        try:
            x = fm.solve(expected_mu)
            denom = float(np.sqrt(max(0.0, float(np.dot(expected_mu, x)))))
        except (np.linalg.LinAlgError, ValueError) as exc:
            _logger.warning("Woodbury solve failed at t=%s: %s", t, exc)
            return np.zeros_like(expected_mu), SolveStatus.DEGENERATE
        if not np.isfinite(denom) or denom <= denom_tol:
            _logger.warning(
                "Positions zeroed at t=%s (sliding_window): normalisation "
                "denominator is degenerate (denom=%s, denom_tol=%s).",
                t,
                denom,
                denom_tol,
            )
            return np.zeros_like(expected_mu), SolveStatus.DEGENERATE
        return x / denom, SolveStatus.VALID

    def _replay_profit_variance(
        self: _EngineProtocol,
        risk_pos_np: np.ndarray,
        cash_pos_np: np.ndarray,
        vola_np: np.ndarray,
        returns_num: np.ndarray,
    ) -> float:
        """Replay the profit-variance EMA across all rows, filling position arrays.

        Iterates :meth:`_iter_solve` row-by-row, updating *profit_variance* and
        writing scaled positions into the pre-allocated arrays **in place**.

        Mutation contract:

        * ``risk_pos_np[i, mask] = pos / profit_variance`` for every solved
          row.  ``WARMUP`` rows (``pos is None``) are never written; those
          cells retain their initial ``NaN``.
        * ``cash_pos_np`` is updated in two passes per row.  At row *i* the
          masked positions are set via :meth:`_scale_to_cash`.  On the *next*
          iteration the full previous row is overwritten as
          ``cash_pos_np[i] = risk_pos_np[i] / vola_np[i]`` to obtain
          PnL-consistent cash positions for the profit update.  Rows
          ``0 … T-2`` therefore reflect this full-row overwrite; only
          ``cash_pos_np[T-1]`` retains the masked write.  Callers that read
          only ``cash_pos_np[-1]`` (e.g. :meth:`warmup_state`) are not
          affected by this distinction.

        Args:
            risk_pos_np: Pre-allocated ``(T, N)`` array for risk positions.
            cash_pos_np: Pre-allocated ``(T, N)`` array for cash positions.
            vola_np: ``(T, N)`` EWMA volatility array.
            returns_num: ``(T, N)`` percentage-return array (row 0 is zeros).

        Returns:
            float: Final EWMA profit-variance scalar after processing all rows.
        """
        profit_variance: float = self.cfg.profit_variance_init
        lamb = self.cfg.profit_variance_decay
        for i, _t, mask, pos, _status in self._iter_solve():
            if i > 0:
                ret_mask = np.isfinite(returns_num[i]) & mask
                if ret_mask.any():
                    with np.errstate(invalid="ignore"):
                        cash_pos_np[i - 1] = risk_pos_np[i - 1] / vola_np[i - 1]
                    lhs = np.nan_to_num(cash_pos_np[i - 1, ret_mask], nan=0.0)
                    rhs = np.nan_to_num(returns_num[i, ret_mask], nan=0.0)
                    profit = float(lhs @ rhs)
                    profit_variance = lamb * profit_variance + (1 - lamb) * profit**2
            if pos is not None:
                risk_pos_np[i, mask] = pos / profit_variance
                cash_pos_np[i, mask] = _SolveMixin._scale_to_cash(pos, profit_variance, vola_np[i, mask])
        return profit_variance

    def _iter_matrices(self: _EngineProtocol):
        r"""Yield ``(i, t, mask, matrix)`` for every timestamp.

        ``matrix`` is the effective :math:`(n_{\text{sub}},\ n_{\text{sub}})`
        correlation matrix for the active assets (those with finite prices at
        timestamp *t*).  Yields ``None`` when no valid matrix is available
        (e.g., before the warm-up period has elapsed or when no assets have
        finite prices).

        The behaviour depends on :attr:`BasanosConfig.covariance_config`:

        * :class:`EwmaShrinkConfig`:  Applies :func:`~basanos.math._signal.shrink2id` to
          the EWMA correlation matrix (same computation as
          :attr:`cash_position`).
        * :class:`SlidingWindowConfig`: Builds a
          :class:`~basanos.math._factor_model.FactorModel` from the last
          ``cfg.covariance_config.window`` rows of vol-adjusted returns and returns its
          :attr:`~basanos.math._factor_model.FactorModel.covariance`.

        Yields:
            tuple: ``(i, t, mask, matrix)`` where

            * ``i`` (*int*): Row index into ``self.prices``.
            * ``t``: Timestamp value from ``self.prices["date"]``.
            * ``mask`` (*np.ndarray[bool]*): Shape ``(n_assets,)``; ``True``
              for assets with finite prices at row *i*.
            * ``matrix`` (*np.ndarray | None*): Shape
              ``(mask.sum(), mask.sum())`` or ``None``.
        """
        assets = self.assets
        prices_num = self.prices.select(assets).to_numpy()
        dates = self.prices["date"].to_list()

        if isinstance(self.cfg.covariance_config, EwmaShrinkConfig):
            cor = self.cor
            for i, t in enumerate(dates):
                mask = _SolveMixin._compute_mask(prices_num[i])
                if not mask.any():
                    yield i, t, mask, None
                    continue
                corr_n = cor[t]
                matrix = shrink2id(corr_n, lamb=self.cfg.shrink)[np.ix_(mask, mask)]
                yield i, t, mask, matrix
        else:
            sw_config = cast(SlidingWindowConfig, self.cfg.covariance_config)
            win_w: int = sw_config.window
            win_k: int = sw_config.n_factors
            ret_adj_np = self.ret_adj.select(assets).to_numpy()
            for i, t in enumerate(dates):
                mask = _SolveMixin._compute_mask(prices_num[i])
                in_warmup = i + 1 < win_w
                if not mask.any() or in_warmup:
                    yield i, t, mask, None
                    continue
                window_ret = ret_adj_np[i + 1 - win_w : i + 1][:, mask]
                window_ret = np.where(np.isfinite(window_ret), window_ret, 0.0)
                n_sub = int(mask.sum())
                k_eff = min(win_k, win_w, n_sub)
                try:
                    fm = FactorModel.from_returns(window_ret, k=k_eff)
                    yield i, t, mask, fm.covariance
                except (np.linalg.LinAlgError, ValueError) as exc:
                    _logger.warning("Factor model fit failed at t=%s: %s", t, exc)
                    yield i, t, mask, None

    def _iter_solve(self: _EngineProtocol):
        r"""Yield ``(i, t, mask, pos_or_none, status)`` for every timestamp.

        This is the single authoritative implementation of the per-timestamp
        position-determination logic shared by :attr:`cash_position` and
        :attr:`position_status`.  Extracting it here eliminates the DRY
        violation where both properties previously duplicated the mask
        computation, covariance dispatch, :math:`\mu` handling, denominator
        check, and linear-solve logic.

        Yields:
            tuple: ``(i, t, mask, pos_or_none, status)`` where

            * ``i`` (*int*): Row index into ``self.prices``.
            * ``t``: Timestamp value from ``self.prices["date"]``.
            * ``mask`` (*np.ndarray[bool]*): Shape ``(n_assets,)``; ``True``
              for assets with finite prices at row *i*.
            * ``pos_or_none`` (*np.ndarray | None*): Per-active-asset position
              vector **before** ``profit_variance`` scaling.  The value and its
              downstream effect depend on ``status`` as follows:

              .. list-table::
                 :header-rows: 1

                 * - ``status``
                   - ``pos_or_none``
                   - Downstream effect in :attr:`cash_position`
                 * - ``'warmup'``
                   - ``None``
                   - Positions stay ``NaN`` — insufficient history.
                 * - ``'zero_signal'``
                   - ``np.zeros(n_active)``
                   - Positions written as ``0`` for all active assets.
                 * - ``'degenerate'``
                   - ``np.zeros(n_active)`` (empty when no prices)
                   - Positions written as ``0`` for active assets; rows with
                     no finite prices have an empty mask so all positions
                     remain ``NaN`` as a natural consequence.
                 * - ``'valid'``
                   - ``np.ndarray`` of shape ``(n_active,)``
                   - Solved positions written for all active assets.

              ``None`` is yielded **only** for ``'warmup'`` rows.  All other
              statuses yield an ``np.ndarray`` (possibly zero-length when
              ``mask`` is all-``False``), so consumers can branch solely on
              ``pos_or_none is None`` to detect the warmup case without
              inspecting ``status``.

            * ``status`` (*str*): One of ``'warmup'``, ``'zero_signal'``,
              ``'degenerate'``, or ``'valid'``.
        """
        assets = self.assets
        prices_num = self.prices.select(assets).to_numpy()
        mu_np = self.mu.select(assets).to_numpy()
        dates = self.prices["date"].to_list()

        if isinstance(self.cfg.covariance_config, EwmaShrinkConfig):
            cor = self.cor
            for i, t in enumerate(dates):
                mask = _SolveMixin._compute_mask(prices_num[i])
                if not mask.any():
                    yield i, t, mask, np.zeros(0), SolveStatus.DEGENERATE
                    continue
                expected_mu, sig_status = _SolveMixin._prepare_mu(mu_np[i], mask)
                if sig_status is not None:
                    yield i, t, mask, np.zeros_like(expected_mu), sig_status
                    continue
                corr_n = cor[t]
                matrix = shrink2id(corr_n, lamb=self.cfg.shrink)[np.ix_(mask, mask)]
                pos, status = _SolveMixin._solve_ewma_step(expected_mu, matrix, t, self.cfg.denom_tol)
                yield i, t, mask, pos, status
        else:
            sw_config = cast(SlidingWindowConfig, self.cfg.covariance_config)
            win_w: int = sw_config.window
            win_k: int = sw_config.n_factors
            ret_adj_np = self.ret_adj.select(assets).to_numpy()
            for i, t in enumerate(dates):
                mask = _SolveMixin._compute_mask(prices_num[i])
                if not mask.any():
                    yield i, t, mask, np.zeros(0), SolveStatus.DEGENERATE
                    continue
                in_warmup = i + 1 < win_w
                if in_warmup:
                    yield i, t, mask, None, SolveStatus.WARMUP
                    continue
                window_ret = ret_adj_np[i + 1 - win_w : i + 1][:, mask]
                window_ret = np.where(np.isfinite(window_ret), window_ret, 0.0)
                n_sub = int(mask.sum())
                k_eff = min(win_k, win_w, n_sub)
                try:
                    fm = FactorModel.from_returns(window_ret, k=k_eff)
                except (np.linalg.LinAlgError, ValueError) as exc:
                    _logger.warning("Sliding window SVD failed at t=%s: %s", t, exc)
                    yield i, t, mask, np.zeros(n_sub), SolveStatus.DEGENERATE
                    continue
                expected_mu, sig_status = _SolveMixin._prepare_mu(mu_np[i], mask)
                if sig_status is not None:
                    yield i, t, mask, np.zeros_like(expected_mu), sig_status
                    continue
                pos, status = _SolveMixin._solve_sw_step(expected_mu, fm, t, self.cfg.denom_tol)
                yield i, t, mask, pos, status

    def warmup_state(self: _EngineProtocol) -> WarmupState:
        """Return the final :class:`WarmupState` after replaying the full batch.

        Encapsulates the profit-variance EMA replay loop that was previously
        duplicated inside :meth:`BasanosStream.from_warmup`.  By centralising
        the loop here, :meth:`~BasanosStream.from_warmup` no longer needs to
        call the private :meth:`_iter_solve` generator directly.

        Returns:
            WarmupState: A frozen dataclass with:

            * ``profit_variance`` - final EWMA profit-variance scalar.
            * ``prev_cash_pos`` - cash-position vector at the last row,
              shape ``(n_assets,)``.

        Examples:
            >>> import numpy as np
            >>> import polars as pl
            >>> from basanos.math import BasanosConfig, BasanosEngine
            >>> rng = np.random.default_rng(0)
            >>> dates = list(range(30))
            >>> prices = pl.DataFrame({
            ...     "date": dates,
            ...     "A": np.cumprod(1 + rng.normal(0.001, 0.02, 30)) * 100.0,
            ...     "B": np.cumprod(1 + rng.normal(0.001, 0.02, 30)) * 150.0,
            ... })
            >>> mu = pl.DataFrame({
            ...     "date": dates,
            ...     "A": rng.normal(0, 0.5, 30),
            ...     "B": rng.normal(0, 0.5, 30),
            ... })
            >>> cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)
            >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
            >>> ws = engine.warmup_state()
            >>> isinstance(ws.profit_variance, float)
            True
            >>> ws.prev_cash_pos.shape
            (2,)
        """
        assets = self.assets
        n_assets = len(assets)
        n_rows = self.prices.height
        prices_np = self.prices.select(assets).to_numpy()
        vola_np = self.vola.select(assets).to_numpy()

        returns_num = np.zeros((n_rows, n_assets), dtype=float)
        if n_rows > 1:
            returns_num[1:] = prices_np[1:] / prices_np[:-1] - 1.0

        risk_pos_np = np.full((n_rows, n_assets), np.nan, dtype=float)
        cash_pos_np = np.full((n_rows, n_assets), np.nan, dtype=float)

        if isinstance(self.cfg.covariance_config, EwmaShrinkConfig):
            # Compute the IIR filter state in a single pass over the warmup data
            # so BasanosStream.from_warmup() can seed the incremental lfilter
            # without a second sweep.
            ret_adj_np = self.ret_adj.select(assets).to_numpy()
            _, iir_state = _ewm_corr_with_final_state(
                ret_adj_np,
                com=self.cfg.corr,
                min_periods=self.cfg.corr,
                min_corr_denom=self.cfg.min_corr_denom,
            )
        else:
            iir_state = None

        profit_variance = _SolveMixin._replay_profit_variance(self, risk_pos_np, cash_pos_np, vola_np, returns_num)
        prev_cash_pos = cash_pos_np[-1].copy()
        return WarmupState(
            profit_variance=profit_variance,
            prev_cash_pos=prev_cash_pos,
            corr_iir_state=iir_state,
        )
