"""Incremental (streaming) API for BasanosEngine.

This private module defines three public symbols:

* :class:`_StreamState` — mutable dataclass that persists all O(N²) IIR
  filter and EWMA accumulator state between consecutive
  :meth:`BasanosStream.step` calls.  Kept separate from the engine so the
  state layout can be read and tested in isolation.
* :class:`StepResult` — frozen dataclass returned by each
  :meth:`BasanosStream.step` call.
* :class:`BasanosStream` — incremental façade with a
  :meth:`~BasanosStream.from_warmup` classmethod and a
  :meth:`~BasanosStream.step` method.

IIR state model
---------------
The EWM recurrence ``s[t] = beta·s[t-1] + v[t]`` is a causal, single-pole IIR
filter.  When the full history is available, ``scipy.signal.lfilter`` solves
all N² pairs in one vectorised call and discards the intermediate array.

In the *incremental* setting there is no history — only the current sample
arrives at each ``step()``.  To continue the same recurrence across calls we
need the *filter memory*, i.e. the value of the accumulator at the end of the
previous call.

``scipy.signal.lfilter`` exposes this directly: when called as::

    y, zf = lfilter(b, a, x, zi=zi)

``zi`` is the initial state (shape ``(max(len(a), len(b)) - 1, …)`` = ``(1, …)``
for our first-order filter) and ``zf`` is the *final* state after processing
``x``.  Passing the returned ``zf`` back as ``zi`` in the next call is
mathematically equivalent to having run ``lfilter`` over the concatenated
input, so the incremental and batch paths produce bit-for-bit identical
results.

The four correlation accumulators (``corr_zi_x``, ``corr_zi_x2``,
``corr_zi_xy``, ``corr_zi_w``) follow exactly this pattern.  Each has shape
``(1, N, N)`` — the leading 1 is the IIR filter order required by
``lfilter``'s ``zi`` argument.

The volatility accumulators (``vola_*``, ``pct_*``) use a simpler scalar
recurrence and store the running sums directly as ``(N,)`` arrays.

Memory
------
Total incremental state is 4x(1,N,N) + (N,N) + 8x(N,) + O(1) scalars,
giving **O(N^2)** memory independent of the number of timesteps processed.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np
import polars as pl
from scipy.signal import lfilter

from ..exceptions import MissingDateColumnError
from ._config import BasanosConfig, EwmaShrinkConfig
from ._linalg import inv_a_norm, solve
from ._signal import shrink2id

_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _StreamState:
    """Mutable state carrier for one :class:`BasanosStream` instance.

    All arrays are updated in-place (or replaced) by ``BasanosStream.step()``.
    The class is intentionally *not* frozen so that the step method can modify
    fields directly without creating a new object on every tick.

    IIR filter state (correlation)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The four ``corr_zi_*`` fields are the *final conditions* (``zf``) returned
    by ``scipy.signal.lfilter`` after the previous step.  They are passed back
    as the ``zi`` argument on the next step so that the incremental filter
    produces the same numerical result as a single batch call over all history.
    See the module docstring for the full derivation.

    ``beta_corr = cfg.corr / (1 + cfg.corr)``  (from ``com = cfg.corr``)

    EWM accumulator state (volatility)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``vola_*`` and ``pct_*`` accumulate the running weighted sums needed to
    compute exponentially-weighted standard deviations:

    * ``s_x``  — EWM sum of x (numerator of the mean)
    * ``s_x2`` — EWM sum of x² (numerator of the second moment)
    * ``s_w``  — EWM sum of weights (denominator)
    * ``s_w2`` — EWM sum of squared weights (for bias correction)

    ``beta_vola = (cfg.vola - 1) / cfg.vola``  (from ``com = cfg.vola - 1``)

    Attributes:
        corr_zi_x:  IIR filter state for the x-accumulator of the EWM
            correlation; shape ``(1, N, N)``.
        corr_zi_x2: IIR filter state for the x²-accumulator; shape
            ``(1, N, N)``.
        corr_zi_xy: IIR filter state for the xy-accumulator; shape
            ``(1, N, N)``.
        corr_zi_w:  IIR filter state for the weight-accumulator; shape
            ``(1, N, N)``.
        corr_count: Cumulative joint-finite observation count per asset pair;
            shape ``(N, N)`` dtype int.
        vola_s_x:   EWM sum of volatility-adjusted log-returns; shape ``(N,)``.
        vola_s_x2:  EWM sum of squared vol-adj log-returns; shape ``(N,)``.
        vola_s_w:   EWM weight sum for vol accumulators; shape ``(N,)``.
        vola_s_w2:  EWM squared-weight sum for vol accumulators; shape ``(N,)``.
        vola_count: Cumulative finite observation count for vol; shape ``(N,)``
            dtype int.
        pct_s_x:    EWM sum of pct-returns; shape ``(N,)``.
        pct_s_x2:   EWM sum of squared pct-returns; shape ``(N,)``.
        pct_s_w:    EWM weight sum for pct accumulators; shape ``(N,)``.
        pct_s_w2:   EWM squared-weight sum for pct accumulators; shape ``(N,)``.
        pct_count:  Cumulative finite observation count for pct-return vol;
            shape ``(N,)`` dtype int.
        profit_variance: EMA of squared realised profit; initialised to
            ``cfg.profit_variance_init``.
        prev_price:    Last price row seen, used to compute returns on the next
            step; shape ``(N,)``.
        prev_cash_pos: Last cash position, used to compute profit on the next
            step; shape ``(N,)``.
        step_count: Number of steps processed so far (0 before first step).
    """

    # ── IIR filter state for EWM correlation — shape (1, N, N) each ──────────
    corr_zi_x: np.ndarray  # (1, N, N)
    corr_zi_x2: np.ndarray  # (1, N, N)
    corr_zi_xy: np.ndarray  # (1, N, N)
    corr_zi_w: np.ndarray  # (1, N, N)
    corr_count: np.ndarray  # (N, N) int — cumulative joint-finite observation count

    # ── EWMA accumulators for vol_adj (log-return std; com=vola-1, min_samples=1) ──
    vola_s_x: np.ndarray  # (N,)
    vola_s_x2: np.ndarray  # (N,)
    vola_s_w: np.ndarray  # (N,)
    vola_s_w2: np.ndarray  # (N,)
    vola_count: np.ndarray  # (N,) int

    # ── EWMA accumulators for vola (pct-return std; com=vola-1, min_samples=vola) ──
    pct_s_x: np.ndarray  # (N,)
    pct_s_x2: np.ndarray  # (N,)
    pct_s_w: np.ndarray  # (N,)
    pct_s_w2: np.ndarray  # (N,)
    pct_count: np.ndarray  # (N,) int

    # ── Scalars ───────────────────────────────────────────────────────────────
    profit_variance: float  # EMA of squared profit; initialised to cfg.profit_variance_init
    prev_price: np.ndarray  # (N,) last price row (to compute returns at next step)
    prev_cash_pos: np.ndarray  # (N,) last cash position (to compute profit at next step)
    step_count: int


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
# Helper: unbiased EWMA std from running accumulators
# ---------------------------------------------------------------------------


def _ewm_std_from_state(
    s_x: np.ndarray,
    s_x2: np.ndarray,
    s_w: np.ndarray,
    s_w2: np.ndarray,
    count: np.ndarray,
    min_samples: int,
) -> np.ndarray:
    r"""Compute the unbiased EWMA standard deviation from running accumulators.

    Implements the same Bessel-corrected formula used by
    ``polars.Expr.ewm_std(adjust=True)``::

        var_biased  = s_x2/s_w - (s_x/s_w)^2
        correction  = s_w^2 / (s_w^2 - s_w2)      # Bessel correction
        var_unbiased = var_biased * correction
        std          = sqrt(max(0, var_unbiased))

    where ``s_w2 = sum(wi^2)`` is the sum of squared EWM weights.

    Parameters
    ----------
    s_x, s_x2, s_w, s_w2:
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
    n = len(s_x)
    result = np.full(n, np.nan, dtype=float)
    ok = count >= min_samples
    if not ok.any():
        return result

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(s_w > 0, s_x / s_w, 0.0)
        mean_sq = np.where(s_w > 0, s_x2 / s_w, 0.0)
        var_biased = np.maximum(mean_sq - mean**2, 0.0)
        denom_corr = s_w**2 - s_w2
        # denom_corr > 0 iff count >= 2; equals 0 when count == 1
        var_unbiased = np.where(denom_corr > 0, var_biased * s_w**2 / denom_corr, 0.0)
        std = np.sqrt(var_unbiased)

    return np.where(ok, std, np.nan)


# ---------------------------------------------------------------------------
# BasanosStream
# ---------------------------------------------------------------------------


class BasanosStream:
    """Incremental (streaming) optimiser backed by a single :class:`_StreamState`.

    After warming up on a historical batch via :meth:`from_warmup`, each call
    to :meth:`step` advances the internal state by exactly one row in
    O(N^2) time — without revisiting the full warmup history.

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

    _cfg: BasanosConfig
    _assets: list[str]
    _state: _StreamState

    def __init__(self, cfg: BasanosConfig, assets: list[str], state: _StreamState) -> None:
        """Initialise from an explicit config, asset list, and state container."""
        object.__setattr__(self, "_cfg", cfg)
        object.__setattr__(self, "_assets", assets)
        object.__setattr__(self, "_state", state)

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent accidental attribute mutation — BasanosStream is immutable."""
        raise dataclasses.FrozenInstanceError(f"{type(self).__name__}.{name}")

    @property
    def assets(self) -> list[str]:
        """Ordered list of asset column names."""
        return self._assets

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
        advances the optimiser in O(N^2) time without touching the warmup
        data again.

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
        # The lfilter zi format requires shape (1, N, N) for a first-order filter.
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
        zi0 = np.zeros((1, n_assets, n_assets))
        # lfilter returns (y, zf); zf has shape (1, n_assets, n_assets) and is
        # the final filter state — pass back as zi on the next step.
        _, corr_zi_x = lfilter([1.0], filt_a_corr, v_x, axis=0, zi=zi0)
        _, corr_zi_x2 = lfilter([1.0], filt_a_corr, v_x2, axis=0, zi=zi0)
        _, corr_zi_xy = lfilter([1.0], filt_a_corr, v_xy, axis=0, zi=zi0)
        _, corr_zi_w = lfilter([1.0], filt_a_corr, v_w, axis=0, zi=zi0)
        corr_count: np.ndarray = np.sum(joint_fin.astype(np.int64), axis=0)  # (n_assets, n_assets)

        # 4. Derive EWMA volatility accumulators (vectorised) ---------------
        # Both log-return (for vol_adj) and pct-return (for vola) use the
        # same beta = (vola-1)/vola.  NaN observations (leading NaN at row 0
        # from diff/pct_change) are skipped — the filter input is 0 for NaN
        # rows and the weight accumulator (s_w) only increments for finite
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

        vola_s_x: np.ndarray = lfilter([1.0], filt_vola_a, log_ret_z, axis=0)[-1]
        vola_s_x2: np.ndarray = lfilter([1.0], filt_vola_a, log_ret_z**2, axis=0)[-1]
        # s_w/s_w2 increment only for finite obs (same as ignore_nulls=True for
        # leading-NaN data, which covers all typical price series)
        vola_s_w: np.ndarray = lfilter([1.0], filt_vola_a, fin_log, axis=0)[-1]
        vola_s_w2: np.ndarray = lfilter([1.0], filt_vola2_a, fin_log, axis=0)[-1]
        vola_count: np.ndarray = fin_log.sum(axis=0).astype(int)

        pct_s_x: np.ndarray = lfilter([1.0], filt_vola_a, pct_ret_z, axis=0)[-1]
        pct_s_x2: np.ndarray = lfilter([1.0], filt_vola_a, pct_ret_z**2, axis=0)[-1]
        pct_s_w: np.ndarray = lfilter([1.0], filt_vola_a, fin_pct, axis=0)[-1]
        pct_s_w2: np.ndarray = lfilter([1.0], filt_vola2_a, fin_pct, axis=0)[-1]
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
        prev_price: np.ndarray = prices_np[-1].copy()

        # 6. Construct _StreamState and return ------------------------------
        state = _StreamState(
            corr_zi_x=corr_zi_x,
            corr_zi_x2=corr_zi_x2,
            corr_zi_xy=corr_zi_xy,
            corr_zi_w=corr_zi_w,
            corr_count=corr_count,
            vola_s_x=vola_s_x,
            vola_s_x2=vola_s_x2,
            vola_s_w=vola_s_w,
            vola_s_w2=vola_s_w2,
            vola_count=vola_count,
            pct_s_x=pct_s_x,
            pct_s_x2=pct_s_x2,
            pct_s_w=pct_s_w,
            pct_s_w2=pct_s_w2,
            pct_count=pct_count,
            profit_variance=profit_variance,
            prev_price=prev_price,
            prev_cash_pos=prev_cash_pos,
            step_count=n_rows,
        )
        return cls(cfg=cfg, assets=assets, state=state)

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

        cfg = self._cfg
        assets = self._assets
        state = self._state
        n_assets = len(assets)

        # ── Check if still in the EWM warmup period ─────────────────────────
        # step_count is initialised to n_rows in from_warmup, so this is True
        # for the first (cfg.corr - n_rows) calls when the warmup batch was
        # shorter than cfg.corr (i.e. not enough rows to populate the EWM
        # correlation matrix).  All accumulators are still updated so that the
        # state is ready the moment the warmup period ends.
        in_warmup: bool = state.step_count < cfg.corr  # insufficient samples for corr matrix

        # ── Resolve inputs to (N,) float64 arrays ──────────────────────────
        if isinstance(new_prices, dict):
            new_p = np.array([float(new_prices[a]) for a in assets], dtype=float)
        else:
            new_p = np.asarray(new_prices, dtype=float).ravel()

        if isinstance(new_mu, dict):
            new_m = np.array([float(new_mu[a]) for a in assets], dtype=float)
        else:
            new_m = np.asarray(new_mu, dtype=float).ravel()

        prev_p = state.prev_price
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
        vola_s_x = beta_vola * state.vola_s_x + np.where(fin_log, log_ret, 0.0)
        vola_s_x2 = beta_vola * state.vola_s_x2 + np.where(fin_log, log_ret**2, 0.0)
        vola_s_w = beta_vola * state.vola_s_w + fin_log.astype(float)
        vola_s_w2 = beta_vola_sq * state.vola_s_w2 + fin_log.astype(float)
        vola_count = state.vola_count + fin_log.astype(int)

        # ── Update pct-return EWMA accumulators ────────────────────────────
        fin_pct = np.isfinite(pct_ret)
        pct_s_x = beta_vola * state.pct_s_x + np.where(fin_pct, pct_ret, 0.0)
        pct_s_x2 = beta_vola * state.pct_s_x2 + np.where(fin_pct, pct_ret**2, 0.0)
        pct_s_w = beta_vola * state.pct_s_w + fin_pct.astype(float)
        pct_s_w2 = beta_vola_sq * state.pct_s_w2 + fin_pct.astype(float)
        pct_count = state.pct_count + fin_pct.astype(int)

        # ── Compute vol-adjusted return (for the correlation IIR input) ─────
        log_vol = _ewm_std_from_state(vola_s_x, vola_s_x2, vola_s_w, vola_s_w2, vola_count, min_samples=1)
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

        new_v_x = (va_f[:, np.newaxis] * joint_fin)[np.newaxis]  # (1, N, N)
        new_v_x2 = ((va_f**2)[:, np.newaxis] * joint_fin)[np.newaxis]  # (1, N, N)
        new_v_xy = (va_f[:, np.newaxis] * va_f[np.newaxis, :])[np.newaxis]  # (1, N, N)
        new_v_w = joint_fin.astype(np.float64)[np.newaxis]  # (1, N, N)

        filt_a_corr = np.array([1.0, -beta_corr])
        # y_x[0] is the current-step EWM state (filter output); corr_zi_x is the
        # new filter memory (zf = beta * y[0]) to pass as zi on the next step.
        y_x, corr_zi_x = lfilter([1.0], filt_a_corr, new_v_x, axis=0, zi=state.corr_zi_x)
        y_x2, corr_zi_x2 = lfilter([1.0], filt_a_corr, new_v_x2, axis=0, zi=state.corr_zi_x2)
        y_xy, corr_zi_xy = lfilter([1.0], filt_a_corr, new_v_xy, axis=0, zi=state.corr_zi_xy)
        y_w, corr_zi_w = lfilter([1.0], filt_a_corr, new_v_w, axis=0, zi=state.corr_zi_w)
        corr_count = state.corr_count + joint_fin.astype(np.int64)

        # ── Early return during EWM warmup period ───────────────────────────
        # All accumulators are already updated above; skip the O(N²) matrix
        # reconstruction and O(N³) Cholesky solve which are wasteful during
        # warmup — the computed positions would be discarded anyway.
        if in_warmup:
            state.corr_zi_x = corr_zi_x
            state.corr_zi_x2 = corr_zi_x2
            state.corr_zi_xy = corr_zi_xy
            state.corr_zi_w = corr_zi_w
            state.corr_count = corr_count
            state.vola_s_x = vola_s_x
            state.vola_s_x2 = vola_s_x2
            state.vola_s_w = vola_s_w
            state.vola_s_w2 = vola_s_w2
            state.vola_count = vola_count
            state.pct_s_x = pct_s_x
            state.pct_s_x2 = pct_s_x2
            state.pct_s_w = pct_s_w
            state.pct_s_w2 = pct_s_w2
            state.pct_count = pct_count
            state.prev_price = new_p.copy()
            state.step_count += 1
            return StepResult(
                date=date,
                cash_position=np.full(n_assets, np.nan),
                status="warmup",
                vola=np.full(n_assets, np.nan),
            )

        # ── Reconstruct the EWM correlation matrix ─────────────────────────
        # Use y_*[0] (the filter OUTPUT for this step), not zf[0].
        s_x = y_x[0]
        s_x2 = y_x2[0]
        s_xy = y_xy[0]
        s_w = y_w[0]

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
        vola_vec = _ewm_std_from_state(pct_s_x, pct_s_x2, pct_s_w, pct_s_w2, pct_count, min_samples=cfg.vola)

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
        state.corr_zi_x = corr_zi_x
        state.corr_zi_x2 = corr_zi_x2
        state.corr_zi_xy = corr_zi_xy
        state.corr_zi_w = corr_zi_w
        state.corr_count = corr_count
        state.vola_s_x = vola_s_x
        state.vola_s_x2 = vola_s_x2
        state.vola_s_w = vola_s_w
        state.vola_s_w2 = vola_s_w2
        state.vola_count = vola_count
        state.pct_s_x = pct_s_x
        state.pct_s_x2 = pct_s_x2
        state.pct_s_w = pct_s_w
        state.pct_s_w2 = pct_s_w2
        state.pct_count = pct_count
        state.prev_price = new_p.copy()
        state.prev_cash_pos = new_cash_pos.copy()
        state.profit_variance = profit_variance
        state.step_count += 1

        return StepResult(
            date=date,
            cash_position=new_cash_pos,
            status=status,
            vola=vola_vec,
        )
