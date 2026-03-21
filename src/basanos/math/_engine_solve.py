"""Solve/position mixin for BasanosEngine.

This private module contains :class:`_SolveMixin`, which provides the
``_iter_matrices`` and ``_iter_solve`` generator methods.  Separating them
from :mod:`basanos.math.optimizer` keeps the engine facade lean and makes
the per-timestamp solve logic independently readable and testable.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
from collections.abc import Generator
from enum import StrEnum
from typing import TYPE_CHECKING, TypeAlias, cast

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
class MatrixBundle:
    """Container for the covariance matrix and any mode-specific auxiliary state.

    Wrapping the covariance matrix in a dataclass decouples
    :meth:`_SolveMixin._compute_position` from the raw array so that future
    covariance modes (e.g. DCC-GARCH, RMT-cleaned) can carry additional fields
    through the same interface without changing the method signature.

    Attributes:
        matrix: The ``(n_active, n_active)`` covariance sub-matrix for the
            active assets at a given timestamp.
    """

    matrix: np.ndarray


#: Yield type for :meth:`_SolveMixin._iter_matrices`:
#: ``(i, t, mask, bundle)`` where ``bundle`` is ``None`` during warmup/no-data.
MatrixYield: TypeAlias = tuple[int, datetime.date, np.ndarray, MatrixBundle | None]

#: Yield type for :meth:`_SolveMixin._iter_solve`:
#: ``(i, t, mask, pos_or_none, status)`` where ``pos_or_none`` is ``None`` only for warmup rows.
SolveYield: TypeAlias = tuple[int, datetime.date, np.ndarray, np.ndarray | None, SolveStatus]


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
    def _row_early_check(
        i: int,
        t: datetime.date,
        mask: np.ndarray,
        mu_row: np.ndarray,
    ) -> tuple[np.ndarray, SolveYield | None]:
        """Validate the price mask and expected-return signal for a single row.

        Returns an ``(expected_mu, early_yield)`` pair.  When ``early_yield``
        is not ``None``, the caller should ``yield early_yield; continue``
        immediately — the row is either degenerate (empty mask) or has an
        all-zero signal.  When ``early_yield`` is ``None`` the row is ready
        for the mode-specific solve step.

        Args:
            i: Row index.
            t: Timestamp.
            mask: Boolean array of shape ``(n_assets,)`` indicating finite prices.
            mu_row: Expected-return row of shape ``(n_assets,)``.

        Returns:
            tuple: ``(expected_mu, early_yield)`` where ``expected_mu`` is
            ``np.nan_to_num(mu_row[mask])`` and ``early_yield`` is either a
            complete :data:`SolveYield` tuple (when the caller should yield
            and continue) or ``None`` (when the caller should proceed to solve).
        """
        if not mask.any():
            return np.zeros(0), (i, t, mask, np.zeros(0), SolveStatus.DEGENERATE)
        expected_mu = np.nan_to_num(mu_row[mask])
        sig_status = _SolveMixin._check_signal(mu_row, mask)
        if sig_status is not None:
            return expected_mu, (i, t, mask, np.zeros_like(expected_mu), sig_status)
        return expected_mu, None

    @staticmethod
    def _denom_guard_yield(
        i: int,
        t: datetime.date,
        mask: np.ndarray,
        expected_mu: np.ndarray,
        pos_raw: np.ndarray,
        denom: float,
        denom_tol: float,
    ) -> SolveYield:
        """Apply the normalisation-denominator guard and return the appropriate yield tuple.

        Emits a :data:`~logging.WARNING` and returns a
        :attr:`~SolveStatus.DEGENERATE` yield when *denom* is non-finite or at
        or below *denom_tol*; otherwise returns a :attr:`~SolveStatus.VALID`
        yield with normalised positions ``pos_raw / denom``.

        Args:
            i: Row index.
            t: Timestamp.
            mask: Boolean asset mask of shape ``(n_assets,)``.
            expected_mu: Masked expected-return vector of shape ``(n_active,)``.
            pos_raw: Raw (pre-normalisation) position vector of shape ``(n_active,)``.
            denom: Computed normalisation denominator.
            denom_tol: Tolerance threshold below which *denom* is treated as degenerate.

        Returns:
            SolveYield: Either a degenerate or valid ``(i, t, mask, pos, status)`` tuple.
        """
        n_active = len(expected_mu)
        if not np.isfinite(denom) or denom <= denom_tol:
            _logger.warning(
                "Positions zeroed at t=%s: normalisation denominator is degenerate "
                "(denom=%s, denom_tol=%s). Check signal magnitude and covariance matrix.",
                t,
                denom,
                denom_tol,
                extra={
                    "context": {
                        "t": str(t),
                        "denom": denom,
                        "denom_tol": denom_tol,
                    }
                },
            )
            return i, t, mask, np.zeros(n_active), SolveStatus.DEGENERATE
        return i, t, mask, pos_raw / denom, SolveStatus.VALID

    @staticmethod
    def _compute_position(
        i: int,
        t: datetime.date,
        mask: np.ndarray,
        expected_mu: np.ndarray,
        bundle: MatrixBundle,
        denom_tol: float,
    ) -> SolveYield:
        """Shared solve step used by both covariance branches.

        Computes the normalisation denominator via :func:`~basanos.math._linalg.inv_a_norm`
        and solves the linear system via :func:`~basanos.math._linalg.solve`, then
        delegates to :meth:`_denom_guard_yield`.  Handles
        :exc:`~basanos.exceptions.SingularMatrixError` from both calls.

        Accepting a :class:`MatrixBundle` instead of a raw array means future
        covariance modes can attach auxiliary state to the bundle without
        changing this method's signature.

        Args:
            i: Row index.
            t: Timestamp.
            mask: Boolean asset mask of shape ``(n_assets,)``.
            expected_mu: Masked expected-return vector of shape ``(n_active,)``.
            bundle: Covariance bundle whose ``matrix`` field is an
                ``(n_active, n_active)`` covariance matrix for the active assets.
            denom_tol: Tolerance threshold for the normalisation denominator.

        Returns:
            SolveYield: A degenerate or valid ``(i, t, mask, pos, status)`` tuple.
        """
        matrix = bundle.matrix
        try:
            denom = inv_a_norm(expected_mu, matrix)
        except SingularMatrixError:
            denom = float("nan")
        try:
            pos = solve(matrix, expected_mu)
        except SingularMatrixError:
            return i, t, mask, np.zeros_like(expected_mu), SolveStatus.DEGENERATE
        return _SolveMixin._denom_guard_yield(i, t, mask, expected_mu, pos, denom, denom_tol)

    def _replay_profit_variance(
        self: _EngineProtocol,
        risk_pos_np: np.ndarray,
        cash_pos_np: np.ndarray,
        vola_np: np.ndarray,
        returns_num: np.ndarray,
    ) -> float:
        """Replay the profit-variance EMA across all rows, filling position arrays.

        Iterates :meth:`_iter_solve`, writes risk and cash positions into the
        provided pre-allocated arrays, and returns the final
        ``profit_variance`` scalar.  Both arrays are mutated **in-place**.

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

    def _iter_matrices(self: _EngineProtocol) -> Generator[MatrixYield, None, None]:
        r"""Yield ``(i, t, mask, bundle)`` for every timestamp.

        ``bundle`` is a :class:`MatrixBundle` wrapping the effective
        :math:`(n_{\text{sub}},\ n_{\text{sub}})` correlation matrix for the
        active assets (those with finite prices at timestamp *t*).  Yields
        ``None`` when no valid matrix is available (e.g., before the warm-up
        period has elapsed or when no assets have finite prices).

        The behaviour depends on :attr:`BasanosConfig.covariance_config`:

        * :class:`EwmaShrinkConfig`:  Applies :func:`~basanos.math._signal.shrink2id` to
          the EWMA correlation matrix (same computation as
          :attr:`cash_position`).
        * :class:`SlidingWindowConfig`: Builds a
          :class:`~basanos.math._factor_model.FactorModel` from the last
          ``cfg.covariance_config.window`` rows of vol-adjusted returns and returns its
          :attr:`~basanos.math._factor_model.FactorModel.covariance`.

        Yields:
            tuple: ``(i, t, mask, bundle)`` where

            * ``i`` (*int*): Row index into ``self.prices``.
            * ``t``: Timestamp value from ``self.prices["date"]``.
            * ``mask`` (*np.ndarray[bool]*): Shape ``(n_assets,)``; ``True``
              for assets with finite prices at row *i*.
            * ``bundle`` (:class:`MatrixBundle` | ``None``): Covariance bundle
              of shape ``(mask.sum(), mask.sum())``, or ``None``.
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
                yield i, t, mask, MatrixBundle(matrix=matrix)
        else:
            sw_config = cast(SlidingWindowConfig, self.cfg.covariance_config)
            win_w: int = sw_config.window
            win_k: int = sw_config.n_factors
            ret_adj_np = self.ret_adj.select(assets).to_numpy()
            for i, t in enumerate(dates):
                mask = _SolveMixin._compute_mask(prices_num[i])
                if not mask.any() or i + 1 < win_w:
                    yield i, t, mask, None
                    continue
                window_ret = ret_adj_np[i + 1 - win_w : i + 1][:, mask]
                window_ret = np.where(np.isfinite(window_ret), window_ret, 0.0)
                n_sub = int(mask.sum())
                k_eff = min(win_k, win_w, n_sub)
                try:
                    fm = FactorModel.from_returns(window_ret, k=k_eff)
                    yield i, t, mask, MatrixBundle(matrix=fm.covariance)
                except (np.linalg.LinAlgError, ValueError) as exc:
                    _logger.warning("Factor model fit failed at t=%s: %s", t, exc)
                    yield i, t, mask, None

    def _iter_solve(self: _EngineProtocol) -> Generator[SolveYield, None, None]:
        r"""Yield ``(i, t, mask, pos_or_none, status)`` for every timestamp.

        Iterates :meth:`_iter_matrices` for the per-row covariance sub-matrix,
        then applies :meth:`_row_early_check` (mask/signal guard) and
        :meth:`_compute_position` (linear solve and denominator guard).  The two
        covariance modes differ only in how ``matrix`` is built, which
        :meth:`_iter_matrices` already encapsulates.

        * ``matrix is None`` → :attr:`~SolveStatus.WARMUP` (sliding-window before
          sufficient history) or :attr:`~SolveStatus.DEGENERATE` otherwise.
        * Signal all-zero → :attr:`~SolveStatus.ZERO_SIGNAL`.
        * Singular or degenerate solve → :attr:`~SolveStatus.DEGENERATE`.
        * Success → :attr:`~SolveStatus.VALID`.

        Yields:
            SolveYield: ``(i, t, mask, pos_or_none, status)`` — see
            :data:`SolveYield` for detailed field descriptions.
        """
        mu_np = self.mu.select(self.assets).to_numpy()
        is_sw = isinstance(self.cfg.covariance_config, SlidingWindowConfig)
        win_w: int = cast(SlidingWindowConfig, self.cfg.covariance_config).window if is_sw else 0

        for i, t, mask, bundle in self._iter_matrices():
            if bundle is None:
                # Distinguish SW warmup (insufficient history) from no-data / model-failure.
                if is_sw and mask.any() and i + 1 < win_w:
                    yield i, t, mask, None, SolveStatus.WARMUP
                else:
                    yield i, t, mask, np.zeros(int(mask.sum())), SolveStatus.DEGENERATE
                continue
            expected_mu, early = _SolveMixin._row_early_check(i, t, mask, mu_np[i])
            if early is not None:
                yield early
                continue
            yield _SolveMixin._compute_position(i, t, mask, expected_mu, bundle, self.cfg.denom_tol)

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
