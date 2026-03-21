"""Incremental (streaming) API for BasanosEngine.

This private module provides:

* :class:`StepResult` — frozen dataclass returned by each
  :meth:`BasanosStream.step` call.
* :class:`_StreamState` — mutable dataclass capturing all IIR filter state
  needed to advance the optimiser by one row without re-processing history.
* :class:`BasanosStream` — incremental façade with a
  :meth:`~BasanosStream.from_warmup` classmethod and a
  :meth:`~BasanosStream.step` method.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy.signal import lfilter

from ..exceptions import MissingDateColumnError
from ._config import BasanosConfig, EwmaShrinkConfig
from ._linalg import inv_a_norm, solve
from ._signal import shrink2id

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class StepResult:
    """Frozen dataclass representing the output of a single ``BasanosStream`` step.

    Each call to ``BasanosStream.step()`` returns one ``StepResult`` capturing
    the optimised cash positions, the per-asset volatility estimate, the step
    date, and a status label that describes the solver outcome for that
    timestep.

    Attributes:
        date: The timestamp or date label for this step.  The type mirrors
            whatever is stored in the ``'date'`` column of the input prices
            DataFrame (typically a Python :class:`datetime.date`,
            :class:`datetime.datetime`, or a Polars temporal scalar).
        cash_position: Optimised cash-position vector, shape ``(N,)``.
            Entries are ``NaN`` for assets that are still in the EWMA warmup
            period or that are otherwise inactive at this step.
        status: Solver outcome label for this timestep.  One of:

            * ``'warmup'`` — fewer rows have been seen than the EWMA warmup
              requires; all positions are ``NaN``.
            * ``'zero_signal'`` — the expected-return signal vector ``mu`` is
              identically zero; positions are set to zero rather than solved.
            * ``'degenerate'`` — the covariance matrix is ill-conditioned or
              numerically singular; positions cannot be computed reliably and
              are returned as ``NaN``.
            * ``'valid'`` — normal operation; ``cash_position`` holds the
              optimised allocations.
        vola: Per-asset EWMA percentage-return volatility, shape ``(N,)``.
            Values are ``NaN`` during the warmup period before the EWMA has
            accumulated sufficient history.

    Examples:
        >>> import numpy as np
        >>> result = StepResult(
        ...     date="2024-01-02",
        ...     cash_position=np.array([1000.0, -500.0]),
        ...     status="valid",
        ...     vola=np.array([0.012, 0.018]),
        ... )
        >>> result.status
        'valid'
        >>> result.cash_position.shape
        (2,)
    """

    date: object
    cash_position: np.ndarray
    status: str
    vola: np.ndarray


# ---------------------------------------------------------------------------
# Internal state container
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _StreamState:
    """All IIR filter and accumulator state needed for :meth:`BasanosStream.step`.

    This dataclass is an implementation detail of :class:`BasanosStream` and
    is not part of the public API.  Fields are intentionally mutable so that
    :meth:`BasanosStream.step` can update them in-place after each step.

    Shape conventions
    -----------------
    *N* denotes the number of assets.

    Attributes:
        cfg: Immutable engine configuration.
        assets: Ordered list of asset column names.

        corr_s_x: ``(N, N)`` — last IIR filter state for the EWM sum of
            ``x_i`` (volatility-adjusted return of asset *i*) for each pair
            ``(i, j)`` that is jointly finite.
        corr_s_x2: ``(N, N)`` — last IIR state for EWM sum of ``x_i²``.
        corr_s_xy: ``(N, N)`` — last IIR state for EWM sum of ``x_i · x_j``.
        corr_s_w: ``(N, N)`` — last IIR state for the joint-finite weight sum.
        corr_count: ``(N, N)`` int — cumulative count of joint-finite
            observations for each asset pair; used for the ``min_periods``
            check.

        log_s: ``(N,)`` — EWM running sum of log-returns (for vol-adjusted
            return normalisation).
        log_s2: ``(N,)`` — EWM running sum of squared log-returns.
        log_sw: ``(N,)`` — EWM running weight sum (increments only for finite
            observations, matching Polars ``ignore_nulls=False`` semantics for
            a leading-NaN series).
        log_sw2: ``(N,)`` — EWM running sum of squared weights (uses ``β²``
            decay; required for the unbiased Bessel correction).
        log_count: ``(N,)`` int — count of finite log-return observations seen
            so far.

        pct_s: ``(N,)`` — same as ``log_s`` but for percentage returns (used
            for the :attr:`~BasanosEngine.vola` scaling factor).
        pct_s2: ``(N,)`` — EWM running sum of squared pct-returns.
        pct_sw: ``(N,)`` — EWM weight sum for pct-returns.
        pct_sw2: ``(N,)`` — EWM squared-weight sum for pct-returns.
        pct_count: ``(N,)`` int — count of finite pct-return observations.

        prev_prices: ``(N,)`` — price vector from the most recent step (or
            from the last row of the warmup batch).  Used to compute returns
            at the next :meth:`~BasanosStream.step` call.
        prev_cash_pos: ``(N,)`` — cash-position vector from the previous step;
            used to evaluate the portfolio profit that feeds the profit-variance
            EMA update.
        profit_variance: Scalar float — current value of the exponentially
            weighted moving average of squared portfolio profit.  Initialised
            from :attr:`BasanosConfig.profit_variance_init` and updated on
            every step where at least one asset has a finite return.
    """

    cfg: BasanosConfig
    assets: list[str]

    # IIR filter state for EWM correlation (shape: (N, N))
    corr_s_x: np.ndarray
    corr_s_x2: np.ndarray
    corr_s_xy: np.ndarray
    corr_s_w: np.ndarray
    corr_count: np.ndarray  # int64

    # EWMA std accumulators for log-returns
    log_s: np.ndarray
    log_s2: np.ndarray
    log_sw: np.ndarray
    log_sw2: np.ndarray
    log_count: np.ndarray  # int

    # EWMA std accumulators for pct-returns
    pct_s: np.ndarray
    pct_s2: np.ndarray
    pct_sw: np.ndarray
    pct_sw2: np.ndarray
    pct_count: np.ndarray  # int

    # Previous-row anchors
    prev_prices: np.ndarray
    prev_cash_pos: np.ndarray
    profit_variance: float


# ---------------------------------------------------------------------------
# Helper: unbiased EWMA std from running state
# ---------------------------------------------------------------------------


def _ewm_std_from_state(
    s: np.ndarray,
    s2: np.ndarray,
    sw: np.ndarray,
    sw2: np.ndarray,
    count: np.ndarray,
    min_samples: int,
) -> np.ndarray:
    r"""Compute the unbiased EWMA standard deviation from running accumulators.

    Implements the same Bessel-corrected formula used by
    ``polars.Expr.ewm_std(adjust=True)``::

        var_biased  = s2/sw - (s/sw)^2
        correction  = sw^2 / (sw^2 - sw2)      # Bessel correction
        var_unbiased = var_biased * correction
        std          = sqrt(max(0, var_unbiased))

    where ``sw2 = sum(wi^2)`` is the sum of squared EWM weights.

    Parameters
    ----------
    s, s2, sw, sw2:
        Running accumulators, each of shape ``(N,)``.
    count:
        Integer count of finite observations per asset, shape ``(N,)``.
    min_samples:
        Minimum number of finite observations required before returning a
        non-NaN value.

    Returns:
    -------
    np.ndarray of shape ``(N,)`` with per-asset standard deviations.
    NaN is returned for assets where ``count < min_samples``.
    """
    n = len(s)
    result = np.full(n, np.nan, dtype=float)
    ok = count >= min_samples
    if not ok.any():
        return result

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(sw > 0, s / sw, 0.0)
        mean_sq = np.where(sw > 0, s2 / sw, 0.0)
        var_biased = np.maximum(mean_sq - mean**2, 0.0)
        denom_corr = sw**2 - sw2
        # denom_corr > 0 iff count >= 2; equals 0 when count == 1
        var_unbiased = np.where(denom_corr > 0, var_biased * sw**2 / denom_corr, 0.0)
        std = np.sqrt(var_unbiased)

    result = np.where(ok, std, np.nan)
    return result


# ---------------------------------------------------------------------------
# BasanosStream
# ---------------------------------------------------------------------------


class BasanosStream:
    """Incremental (streaming) optimiser backed by a single :class:`_StreamState`.

    After warming up on a historical batch via :meth:`from_warmup`, each call
    to :meth:`step` advances the internal state by exactly one row in
    :math:`O(N^2)` time — without revisiting the full warmup history.

    Attributes:
        assets: Ordered list of asset column names (read-only).

    Examples:
        >>> import numpy as np
        >>> import polars as pl
        >>> from datetime import date, timedelta
        >>> from basanos.math import BasanosConfig, BasanosStream
        >>> rng = np.random.default_rng(0)
        >>> warmup_len = 60
        >>> dates = pl.date_range(
        ...     start=date(2024, 1, 1),
        ...     end=date(2024, 1, 1) + timedelta(days=warmup_len),
        ...     interval="1d",
        ...     eager=True,
        ... )
        >>> prices = pl.DataFrame({
        ...     "date": dates,
        ...     "A": np.cumprod(1 + rng.normal(0.001, 0.02, warmup_len + 1)) * 100.0,
        ...     "B": np.cumprod(1 + rng.normal(0.001, 0.02, warmup_len + 1)) * 150.0,
        ... })
        >>> mu = pl.DataFrame({
        ...     "date": dates,
        ...     "A": rng.normal(0, 0.5, warmup_len + 1),
        ...     "B": rng.normal(0, 0.5, warmup_len + 1),
        ... })
        >>> cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)
        >>> stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
        >>> result = stream.step(
        ...     prices.select(["A", "B"]).to_numpy()[warmup_len],
        ...     mu.select(["A", "B"]).to_numpy()[warmup_len],
        ...     prices["date"][warmup_len],
        ... )
        >>> isinstance(result, StepResult)
        True
        >>> result.cash_position.shape
        (2,)
    """

    def __init__(self, state: _StreamState) -> None:
        self._state = state

    @property
    def assets(self) -> list[str]:
        """Ordered list of asset column names."""
        return self._state.assets

    # ------------------------------------------------------------------
    # from_warmup
    # ------------------------------------------------------------------

    @classmethod
    def from_warmup(
        cls,
        prices: pl.DataFrame,
        mu: pl.DataFrame,
        cfg: BasanosConfig,
    ) -> BasanosStream:
        """Build a :class:`BasanosStream` from a historical warmup batch.

        Runs :class:`~basanos.math.BasanosEngine` on the full warmup batch
        exactly once and extracts the minimal IIR-filter state required for
        subsequent :meth:`step` calls.  After this call, each :meth:`step`
        advances the optimiser in :math:`O(N^2)` time without touching the
        warmup data again.

        Parameters
        ----------
        prices:
            Historical price DataFrame.  Must contain a ``'date'`` column and
            at least one numeric asset column with strictly positive,
            non-monotonic values.
        mu:
            Expected-return signal DataFrame aligned row-by-row with
            ``prices``.
        cfg:
            Engine configuration.  ``cfg.covariance_config`` **must** be an
            :class:`~basanos.math.EwmaShrinkConfig` instance; sliding-window
            mode is not yet supported for streaming.

        Returns:
        -------
        BasanosStream
            A stream instance whose :meth:`step` method is ready to accept the
            row immediately following the last warmup row.

        Raises:
        ------
        TypeError
            If ``cfg.covariance_config`` is not an
            :class:`~basanos.math.EwmaShrinkConfig`.
        MissingDateColumnError
            If ``'date'`` is absent from ``prices``.
        """
        # 1. Validate -------------------------------------------------------
        if not isinstance(cfg.covariance_config, EwmaShrinkConfig):
            _msg = (
                f"BasanosStream.from_warmup() only supports EwmaShrinkConfig; "
                f"got {type(cfg.covariance_config).__name__}. "
                f"SlidingWindowConfig support is deferred."
            )
            raise TypeError(_msg)
        if "date" not in prices.columns:
            raise MissingDateColumnError("prices")

        # 2. Build the engine on the full warmup batch ----------------------
        # Import here to avoid a circular dependency at module level.
        from .optimizer import BasanosEngine

        engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
        assets = engine.assets
        n_assets = len(assets)
        n_rows = prices.height
        prices_np = prices.select(assets).to_numpy()  # (n_rows, n_assets)

        # 3. Extract IIR filter state for EWM correlation -------------------
        # This mirrors the internals of _ewm_corr_numpy exactly.
        beta_corr: float = cfg.corr / (1.0 + cfg.corr)
        ret_adj_np: np.ndarray = engine.ret_adj.select(assets).to_numpy()  # (n_rows, n_assets)

        fin = np.isfinite(ret_adj_np)  # (n_rows, n_assets)
        xt_f = np.where(fin, ret_adj_np, 0.0)  # (n_rows, n_assets)
        joint_fin = fin[:, :, np.newaxis] & fin[:, np.newaxis, :]  # (n_rows, n_assets, n_assets)

        # Per-pair IIR input sequences (identical to _ewm_corr_numpy)
        v_x = xt_f[:, :, np.newaxis] * joint_fin  # (n_rows, n_assets, n_assets)
        v_x2 = (xt_f**2)[:, :, np.newaxis] * joint_fin  # (n_rows, n_assets, n_assets)
        v_xy = xt_f[:, :, np.newaxis] * xt_f[:, np.newaxis, :]  # (n_rows, n_assets, n_assets)
        v_w = joint_fin.astype(np.float64)  # (n_rows, n_assets, n_assets)

        filt_a_corr = np.array([1.0, -beta_corr])
        # lfilter([1], [1, -beta], v, axis=0)[-1] == zf[0] when zi=zeros((1,...))
        corr_s_x: np.ndarray = lfilter([1.0], filt_a_corr, v_x, axis=0)[-1]
        corr_s_x2: np.ndarray = lfilter([1.0], filt_a_corr, v_x2, axis=0)[-1]
        corr_s_xy: np.ndarray = lfilter([1.0], filt_a_corr, v_xy, axis=0)[-1]
        corr_s_w: np.ndarray = lfilter([1.0], filt_a_corr, v_w, axis=0)[-1]
        corr_count: np.ndarray = np.sum(joint_fin.astype(np.int64), axis=0)  # (n_assets, n_assets)

        # 4. Derive EWMA volatility accumulators (vectorised) ---------------
        # Both log-return (for vol_adj) and pct-return (for vola) use the
        # same beta = (vola-1)/vola.  NaN observations (leading NaN at row 0
        # from diff/pct_change) are skipped — the filter input is 0 for NaN
        # rows and the weight accumulator (sw) only increments for finite
        # observations, matching Polars' effective behaviour for a
        # leading-NaN series.
        beta_vola: float = (cfg.vola - 1) / cfg.vola
        beta_vola_sq: float = beta_vola**2

        log_ret = np.full((n_rows, n_assets), np.nan, dtype=float)
        pct_ret = np.full((n_rows, n_assets), np.nan, dtype=float)
        if n_rows > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ret[1:] = np.log(prices_np[1:] / prices_np[:-1])
                pct_ret[1:] = prices_np[1:] / prices_np[:-1] - 1.0

        fin_log = np.isfinite(log_ret).astype(np.float64)  # (n_rows, n_assets)
        log_ret_z = np.where(fin_log.astype(bool), log_ret, 0.0)

        fin_pct = np.isfinite(pct_ret).astype(np.float64)  # (n_rows, n_assets)
        pct_ret_z = np.where(fin_pct.astype(bool), pct_ret, 0.0)

        filt_vola_a = np.array([1.0, -beta_vola])
        filt_vola2_a = np.array([1.0, -beta_vola_sq])

        log_s: np.ndarray = lfilter([1.0], filt_vola_a, log_ret_z, axis=0)[-1]
        log_s2: np.ndarray = lfilter([1.0], filt_vola_a, log_ret_z**2, axis=0)[-1]
        # sw/sw2 increment only for finite obs (same as ignore_nulls=True for
        # leading-NaN data, which covers all typical price series)
        log_sw: np.ndarray = lfilter([1.0], filt_vola_a, fin_log, axis=0)[-1]
        log_sw2: np.ndarray = lfilter([1.0], filt_vola2_a, fin_log, axis=0)[-1]
        log_count: np.ndarray = fin_log.sum(axis=0).astype(int)

        pct_s: np.ndarray = lfilter([1.0], filt_vola_a, pct_ret_z, axis=0)[-1]
        pct_s2: np.ndarray = lfilter([1.0], filt_vola_a, pct_ret_z**2, axis=0)[-1]
        pct_sw: np.ndarray = lfilter([1.0], filt_vola_a, fin_pct, axis=0)[-1]
        pct_sw2: np.ndarray = lfilter([1.0], filt_vola2_a, fin_pct, axis=0)[-1]
        pct_count: np.ndarray = fin_pct.sum(axis=0).astype(int)

        # 5. Replay profit_variance -----------------------------------------
        # Replicate the exact loop from BasanosEngine.cash_position
        # (optimizer.py lines 451-469) to arrive at the correct final
        # profit_variance and prev_cash_pos.
        returns_num = np.zeros((n_rows, n_assets), dtype=float)
        if n_rows > 1:
            returns_num[1:] = prices_np[1:] / prices_np[:-1] - 1.0

        vola_np: np.ndarray = engine.vola.select(assets).to_numpy()  # (n_rows, n_assets)
        profit_variance: float = cfg.profit_variance_init
        lamb: float = cfg.profit_variance_decay

        risk_pos_np = np.full((n_rows, n_assets), np.nan, dtype=float)
        cash_pos_np = np.full((n_rows, n_assets), np.nan, dtype=float)

        for i, _t, mask, pos, _status in engine._iter_solve():
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
                with np.errstate(invalid="ignore"):
                    cash_pos_np[i, mask] = risk_pos_np[i, mask] / vola_np[i, mask]

        prev_cash_pos: np.ndarray = cash_pos_np[-1].copy()
        prev_prices: np.ndarray = prices_np[-1].copy()

        # 6. Construct _StreamState and return ------------------------------
        state = _StreamState(
            cfg=cfg,
            assets=assets,
            corr_s_x=corr_s_x,
            corr_s_x2=corr_s_x2,
            corr_s_xy=corr_s_xy,
            corr_s_w=corr_s_w,
            corr_count=corr_count,
            log_s=log_s,
            log_s2=log_s2,
            log_sw=log_sw,
            log_sw2=log_sw2,
            log_count=log_count,
            pct_s=pct_s,
            pct_s2=pct_s2,
            pct_sw=pct_sw,
            pct_sw2=pct_sw2,
            pct_count=pct_count,
            prev_prices=prev_prices,
            prev_cash_pos=prev_cash_pos,
            profit_variance=profit_variance,
        )
        return cls(state)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        new_prices: np.ndarray | dict[str, float],
        new_mu: np.ndarray | dict[str, float],
        date: Any = None,
    ) -> StepResult:
        """Advance the stream by one row and return the new optimised position.

        Parameters
        ----------
        new_prices:
            Per-asset prices for the new timestep.  Either a numpy array of
            shape ``(N,)`` (assets ordered as in :attr:`assets`) or a dict
            mapping asset names to price values.
        new_mu:
            Per-asset expected-return signals, same format as ``new_prices``.
        date:
            Timestamp for this step (stored in :attr:`StepResult.date`
            verbatim; not used in any computation).

        Returns:
        -------
        StepResult
            Frozen dataclass with ``cash_position``, ``vola``, ``status``, and
            ``date`` for this timestep.
        """
        from ..exceptions import SingularMatrixError

        state = self._state
        cfg = state.cfg
        assets = state.assets
        n_assets = len(assets)

        # ── Resolve inputs to (N,) float64 arrays ──────────────────────────
        if isinstance(new_prices, dict):
            new_p = np.array([float(new_prices[a]) for a in assets], dtype=float)
        else:
            new_p = np.asarray(new_prices, dtype=float).ravel()

        if isinstance(new_mu, dict):
            new_m = np.array([float(new_mu[a]) for a in assets], dtype=float)
        else:
            new_m = np.asarray(new_mu, dtype=float).ravel()

        prev_p = state.prev_prices
        beta_vola: float = (cfg.vola - 1) / cfg.vola
        beta_vola_sq: float = beta_vola**2
        beta_corr: float = cfg.corr / (1.0 + cfg.corr)

        # ── Compute new log-returns and pct-returns ─────────────────────────
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                np.isfinite(new_p) & np.isfinite(prev_p) & (prev_p > 0),
                new_p / prev_p,
                np.nan,
            )
            log_ret = np.log(ratio)
            pct_ret = ratio - 1.0

        # ── Update log-return EWMA accumulators ────────────────────────────
        fin_log = np.isfinite(log_ret)
        log_s = beta_vola * state.log_s + np.where(fin_log, log_ret, 0.0)
        log_s2 = beta_vola * state.log_s2 + np.where(fin_log, log_ret**2, 0.0)
        log_sw = beta_vola * state.log_sw + fin_log.astype(float)
        log_sw2 = beta_vola_sq * state.log_sw2 + fin_log.astype(float)
        log_count = state.log_count + fin_log.astype(int)

        # ── Update pct-return EWMA accumulators ────────────────────────────
        fin_pct = np.isfinite(pct_ret)
        pct_s = beta_vola * state.pct_s + np.where(fin_pct, pct_ret, 0.0)
        pct_s2 = beta_vola * state.pct_s2 + np.where(fin_pct, pct_ret**2, 0.0)
        pct_sw = beta_vola * state.pct_sw + fin_pct.astype(float)
        pct_sw2 = beta_vola_sq * state.pct_sw2 + fin_pct.astype(float)
        pct_count = state.pct_count + fin_pct.astype(int)

        # ── Compute vol-adjusted return (for the correlation IIR input) ─────
        log_vol = _ewm_std_from_state(log_s, log_s2, log_sw, log_sw2, log_count, min_samples=1)
        # Divide; std == 0 yields ±inf → clipped to ±cfg.clip (matches Polars)
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_adj_val = np.where(
                fin_log,
                np.clip(log_ret / log_vol, -cfg.clip, cfg.clip),
                np.nan,
            )

        # ── Update IIR filter state for EWM correlation ─────────────────────
        fin_va = np.isfinite(vol_adj_val)
        va_f = np.where(fin_va, vol_adj_val, 0.0)
        joint_fin = fin_va[:, np.newaxis] & fin_va[np.newaxis, :]  # (N, N)

        v_x = va_f[:, np.newaxis] * joint_fin  # (N, N)
        v_x2 = (va_f**2)[:, np.newaxis] * joint_fin  # (N, N)
        v_xy = va_f[:, np.newaxis] * va_f[np.newaxis, :]  # (N, N)
        v_w = joint_fin.astype(np.float64)  # (N, N)

        s_x = beta_corr * state.corr_s_x + v_x
        s_x2 = beta_corr * state.corr_s_x2 + v_x2
        s_xy = beta_corr * state.corr_s_xy + v_xy
        s_w = beta_corr * state.corr_s_w + v_w
        corr_count = state.corr_count + joint_fin.astype(np.int64)

        # ── Reconstruct the EWM correlation matrix ─────────────────────────
        with np.errstate(divide="ignore", invalid="ignore"):
            pos_w = s_w > 0
            ewm_x = np.where(pos_w, s_x / s_w, np.nan)
            ewm_y = np.where(pos_w, s_x.T / s_w, np.nan)
            ewm_x2 = np.where(pos_w, s_x2 / s_w, np.nan)
            ewm_y2 = np.where(pos_w, s_x2.T / s_w, np.nan)
            ewm_xy = np.where(pos_w, s_xy / s_w, np.nan)

        var_x = np.maximum(ewm_x2 - ewm_x**2, 0.0)
        var_y = np.maximum(ewm_y2 - ewm_y**2, 0.0)
        denom_corr = np.sqrt(var_x * var_y)
        cov = ewm_xy - ewm_x * ewm_y

        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom_corr > cfg.min_corr_denom, cov / denom_corr, np.nan)
        corr = np.clip(corr, -1.0, 1.0)
        corr[corr_count < cfg.corr] = np.nan
        diag_idx = np.arange(n_assets)
        corr[diag_idx, diag_idx] = np.where(corr_count[diag_idx, diag_idx] >= cfg.corr, 1.0, np.nan)

        # ── Apply shrinkage ─────────────────────────────────────────────────
        matrix = shrink2id(corr, lamb=cfg.shrink)

        # ── Compute EWMA volatility (pct-return std) ────────────────────────
        vola_vec = _ewm_std_from_state(pct_s, pct_s2, pct_sw, pct_sw2, pct_count, min_samples=cfg.vola)

        # ── Update profit_variance ──────────────────────────────────────────
        profit_variance = state.profit_variance
        lamb_pv = cfg.profit_variance_decay
        prev_cash_pos = state.prev_cash_pos

        mask = np.isfinite(new_p)
        ret_mask = fin_pct & mask
        if ret_mask.any():
            lhs = np.nan_to_num(prev_cash_pos[ret_mask], nan=0.0)
            rhs = np.nan_to_num(pct_ret[ret_mask], nan=0.0)
            profit = float(lhs @ rhs)
            profit_variance = lamb_pv * profit_variance + (1 - lamb_pv) * profit**2

        # ── Solve for position ───────────────────────────────────────────────
        new_cash_pos = np.full(n_assets, np.nan, dtype=float)
        status = "degenerate"

        if not mask.any():
            status = "degenerate"
        else:
            corr_sub = matrix[np.ix_(mask, mask)]
            expected_mu = np.nan_to_num(new_m[mask])
            if np.allclose(expected_mu, 0.0):
                new_cash_pos[mask] = 0.0
                status = "zero_signal"
            else:
                try:
                    denom_val = inv_a_norm(expected_mu, corr_sub)
                except SingularMatrixError:
                    denom_val = float("nan")

                if not np.isfinite(denom_val) or denom_val <= cfg.denom_tol:
                    _logger.warning(
                        "Positions zeroed at date=%s: normalisation denominator degenerate (denom=%s, denom_tol=%s).",
                        date,
                        denom_val,
                        cfg.denom_tol,
                    )
                    new_cash_pos[mask] = 0.0
                    status = "degenerate"
                else:
                    try:
                        pos = solve(corr_sub, expected_mu) / denom_val
                    except SingularMatrixError:
                        new_cash_pos[mask] = 0.0
                        status = "degenerate"
                    else:
                        risk_pos = pos / profit_variance
                        vola_sub = vola_vec[mask]
                        with np.errstate(invalid="ignore"):
                            new_cash_pos[mask] = risk_pos / vola_sub
                        status = "valid"

        # ── Persist updated state ───────────────────────────────────────────
        self._state = _StreamState(
            cfg=cfg,
            assets=assets,
            corr_s_x=s_x,
            corr_s_x2=s_x2,
            corr_s_xy=s_xy,
            corr_s_w=s_w,
            corr_count=corr_count,
            log_s=log_s,
            log_s2=log_s2,
            log_sw=log_sw,
            log_sw2=log_sw2,
            log_count=log_count,
            pct_s=pct_s,
            pct_s2=pct_s2,
            pct_sw=pct_sw,
            pct_sw2=pct_sw2,
            pct_count=pct_count,
            prev_prices=new_p.copy(),
            prev_cash_pos=new_cash_pos.copy(),
            profit_variance=profit_variance,
        )

        return StepResult(
            date=date,
            cash_position=new_cash_pos,
            status=status,
            vola=vola_vec,
        )
