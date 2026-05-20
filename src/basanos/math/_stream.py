"""Incremental (streaming) API for BasanosEngine.

This private module defines three public symbols:

* `_StreamState` — mutable dataclass that persists all O(N²)
  accumulator state between consecutive `step` calls.  Kept separate
  from the engine so the state layout can be read and tested in isolation.
* `StepResult` — frozen dataclass returned by each
  `step` call.
* `BasanosStream` — incremental façade with a
  `from_warmup` classmethod and a
  `step` method.

EWM correlation state model
----------------------------
In EWM mode the correlation at each step is recomputed by calling
``ewm_covariance`` from ``cvx.linalg`` over the full growing history of
vol-adjusted returns stored in ``corr_ret_buf``.  This keeps the incremental
and batch paths numerically identical at the cost of O(T·N²) time per step
(acceptable for small N or short warmup histories).

The volatility accumulators (``vola_*``, ``pct_*``) use a simpler scalar
recurrence and store the running sums directly as ``(N,)`` arrays.

Memory
------
Total incremental state is O(T·N) for the growing history buffer plus
8x(N,) + O(1) scalars.  For the SlidingWindowConfig the buffer is a fixed
(W, N) array independent of T.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, cast

import numpy as np
import polars as pl
from cvx.linalg import cov_to_corr
from cvx.linalg.ewm_cov import ewm_covariance
from scipy.signal import lfilter

from ..exceptions import MissingDateColumnError, StreamStateCorruptError
from ._config import BasanosConfig, EwmaShrinkConfig, SlidingWindowConfig
from ._engine_solve import MatrixBundle, SolveStatus, _SolveMixin
from ._factor_model import FactorModel
from ._signal import shrink2id

_logger = logging.getLogger(__name__)

#: Increment this when the `save` archive layout changes in
#: a backward-incompatible way.  `load` asserts the stored
#: value matches before deserialising anything, so callers get a clear error
#: instead of a silent ``KeyError`` or wrong state.
_SAVE_FORMAT_VERSION: int = 3


@dataclasses.dataclass
class _StreamState:
    """Mutable state carrier for one `BasanosStream` instance.

    All arrays are updated in-place (or replaced) by ``BasanosStream.step()``.
    The class is intentionally *not* frozen so that the step method can modify
    fields directly without creating a new object on every tick.

    EWM correlation state
    ~~~~~~~~~~~~~~~~~~~~~
    ``corr_ret_buf`` holds the growing history of vol-adjusted returns used by
    ``ewm_covariance`` to recompute the correlation matrix on each step.  It
    is ``None`` for ``SlidingWindowConfig`` (which uses ``sw_ret_buf`` instead).

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
        corr_ret_buf: Growing history of vol-adjusted returns used by
            ``ewm_covariance``; shape ``(T, N)`` for EwmaShrinkConfig.
            ``None`` for SlidingWindowConfig.
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
        prev_price:    Last price row seen, used to compute returns on the next
            step; shape ``(N,)``.
        prev_cash_pos: Last cash position, used to apply the turnover constraint
            on the next step; shape ``(N,)``.
        step_count: Number of steps processed so far (0 before first step).
    """

    # ── EWM correlation history buffer — (T, N) for EwmaShrinkConfig ─────────
    corr_ret_buf: np.ndarray | None  # (T, N) growing history of vol-adj returns; None for SlidingWindowConfig

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
    prev_price: np.ndarray  # (N,) last price row (to compute returns at next step)
    prev_cash_pos: np.ndarray  # (N,) last cash position (for turnover constraint at next step)
    step_count: int

    # ── SlidingWindowConfig state — None for EwmaShrinkConfig ────────────────
    # shape (W, N): last W vol-adjusted returns (oldest row first); None when
    # using EwmaShrinkConfig.  corr_ret_buf above is unused (None) in this mode;
    # sw_ret_buf carries all the correlation state instead.
    sw_ret_buf: np.ndarray | None = None  # (W, N) rolling buffer, or None


#: Keys that `save` writes to the ``.npz`` archive for
#: `_StreamState` fields.  Derived automatically from
#: `fields` so that adding a new field to ``_StreamState``
#: is sufficient — no manual update here is required.
#:
#: The three non-state keys (``format_version``, ``cfg_json``, ``assets``) are
#: added explicitly because they are not fields of ``_StreamState`` itself.
_REQUIRED_KEYS: frozenset[str] = frozenset(
    {f.name for f in dataclasses.fields(_StreamState)} | {"format_version", "cfg_json", "assets"}
)


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
            DataFrame (typically a Python `date`,
            `datetime`, or a Polars temporal scalar).
        cash_position: Optimised cash-position vector, shape ``(N,)``.
            Entries are ``NaN`` for assets that are still in the EWMA warmup
            period or that are otherwise inactive at this step.
        status: Solver outcome label for this timestep
            (`SolveStatus`).  Since `SolveStatus`
            is a ``StrEnum``, values compare equal to their string equivalents
            (e.g. ``result.status == "valid"`` is ``True``):

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
    status: SolveStatus
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
# Helper: batch EWMA volatility accumulators from a returns matrix
# ---------------------------------------------------------------------------


def _ewm_vol_accumulators_from_batch(
    returns: np.ndarray,
    beta: float,
    beta_sq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute final EWMA volatility accumulators from a batch of returns.

    Implements the same IIR recurrence as `step` but
    vectorised over *T* timesteps using ``scipy.signal.lfilter``.  The five
    returned arrays are identical to the accumulators that would result from
    feeding each row of *returns* through the scalar step-by-step recurrence::

        s_x[t]  = beta   * s_x[t-1]  + (x[t] if finite else 0)
        s_x2[t] = beta   * s_x2[t-1] + (x[t]^2 if finite else 0)
        s_w[t]  = beta   * s_w[t-1]  + (1 if finite else 0)
        s_w2[t] = beta^2 * s_w2[t-1] + (1 if finite else 0)

    Parameters
    ----------

    Returns:
        Float array of shape ``(T, N)``.  NaN entries are treated as missing
        observations — they contribute nothing to the numerator sums and do
        not increment the weight accumulators.
    beta:
        EWM decay factor for ``s_x``, ``s_x2``, and ``s_w``
        (``beta = (com) / (1 + com)`` for ``com = cfg.vola - 1``).
    beta_sq:
        Squared decay factor used for ``s_w2``.  Must equal ``beta ** 2``.

    Returns:
    -------
    s_x, s_x2, s_w, s_w2 : np.ndarray of shape ``(N,)``
        Final EWMA running accumulators after processing all *T* rows.
    count : np.ndarray of shape ``(N,)`` dtype int
        Number of finite observations per asset.

    Notes:
    -----
    This function is the shared implementation used by
    `from_warmup` for both the log-return (``vola_*``)
    and pct-return (``pct_*``) accumulators.  Keeping a single implementation
    here guarantees that the batch and incremental paths stay in sync when the
    recurrence definition changes.
    """
    fin = np.isfinite(returns).astype(np.float64)  # (T, N)
    x_z = np.where(fin.astype(bool), returns, 0.0)  # (T, N)
    filt_a = np.array([1.0, -beta])
    filt_a2 = np.array([1.0, -beta_sq])

    s_x: np.ndarray = lfilter([1.0], filt_a, x_z, axis=0)[-1]
    s_x2: np.ndarray = lfilter([1.0], filt_a, x_z**2, axis=0)[-1]
    s_w: np.ndarray = lfilter([1.0], filt_a, fin, axis=0)[-1]
    s_w2: np.ndarray = lfilter([1.0], filt_a2, fin, axis=0)[-1]
    count: np.ndarray = fin.sum(axis=0).astype(int)

    return s_x, s_x2, s_w, s_w2, count


def _resolve_step_vector(
    values: np.ndarray | dict[str, float],
    assets: list[str],
    n_assets: int,
    arg_name: str,
) -> np.ndarray:
    """Resolve one step input to a validated ``(N,)`` float vector.

    Args:
        values: Raw input provided to ``step`` as either dict or array-like.
        assets: Ordered asset names used when ``values`` is a mapping.
        n_assets: Expected vector length.
        arg_name: Argument label used in shape-mismatch errors.

    Returns:
        A float64 numpy vector of shape ``(n_assets,)``.

    Raises:
        ValueError: If the resolved vector does not match ``(n_assets,)``.
    """
    if isinstance(values, dict):
        vector = np.array([float(values[a]) for a in assets], dtype=float)
    else:
        vector = np.asarray(values, dtype=float).ravel()
    if vector.shape != (n_assets,):
        raise ValueError(f"{arg_name} must have shape ({n_assets},); got {vector.shape}")  # noqa: TRY003
    return vector


# ---------------------------------------------------------------------------
# BasanosStream
# ---------------------------------------------------------------------------


class BasanosStream:
    """Incremental (streaming) optimiser backed by a single `_StreamState`.

    After warming up on a historical batch via `from_warmup`, each call
    to `step` advances the internal state by exactly one row in
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
        """Build a `BasanosStream` from a historical warmup batch.

        Runs `BasanosEngine` on the full warmup batch
        exactly once and extracts the minimal IIR-filter state required for
        subsequent `step` calls.  After this call, each `step`
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
            Engine configuration.  Both `EwmaShrinkConfig`
            and `SlidingWindowConfig` are supported.

        Returns:
        -------
        BasanosStream
            A stream instance whose `step` method is ready to accept the
            row immediately following the last warmup row.

        Notes:
        ------
        **Short-warmup behaviour with** ``SlidingWindowConfig``: when
        ``len(prices) < cfg.covariance_config.window``, the internal rolling
        buffer (``sw_ret_buf``) is NaN-padded for the missing prefix rows.
        `step` returns ``StepResult(status="warmup")`` for each of the
        first ``window - len(prices)`` calls, exactly matching the EWM warmup
        semantics.  By the time `step` returns the first non-warmup
        result the buffer contains only real data — no NaN-padded rows remain.

        Raises:
        ------
        MissingDateColumnError
            If ``'date'`` is absent from ``prices``.
        """
        # 1. Validate -------------------------------------------------------
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

        # 3. Extract mode-specific state from WarmupState --------------------
        ws = engine.warmup_state()
        if isinstance(cfg.covariance_config, EwmaShrinkConfig):
            # EWM: seed the growing history buffer from engine.ret_adj so that
            # each subsequent step() can call ewm_covariance over the full history.
            ret_adj_np = engine.ret_adj.select(assets).to_numpy()
            corr_ret_buf: np.ndarray | None = ret_adj_np
            sw_ret_buf: np.ndarray | None = None
        else:
            # SW: carry the last W vol-adjusted returns as a rolling buffer.
            sw_config = cast(SlidingWindowConfig, cfg.covariance_config)
            win_w = sw_config.window
            ret_adj_np = engine.ret_adj.select(assets).to_numpy()  # (n_rows, N)
            if n_rows >= win_w:
                sw_ret_buf = ret_adj_np[-win_w:].copy()
            else:
                sw_ret_buf = np.full((win_w, n_assets), np.nan)
                sw_ret_buf[-n_rows:] = ret_adj_np
            corr_ret_buf = None

        # 4. Derive EWMA volatility accumulators (vectorised) ---------------
        # Both log-return (for vol_adj) and pct-return (for vola) use the
        # same beta = (vola-1)/vola.  NaN observations (leading NaN at row 0
        # from diff/pct_change) are skipped — the filter input is 0 for NaN
        # rows and the weight accumulator (s_w) only increments for finite
        # observations, matching Polars' effective behaviour for a
        # leading-NaN series.
        #
        # Delegate to the shared helper _ewm_vol_accumulators_from_batch so
        # that the batch and incremental recurrences share a single definition.
        beta_vola: float = (cfg.vola - 1) / cfg.vola
        beta_vola_sq: float = beta_vola**2

        log_ret = np.full((n_rows, n_assets), np.nan, dtype=float)
        pct_ret = np.full((n_rows, n_assets), np.nan, dtype=float)
        if n_rows > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_ret[1:] = np.log(prices_np[1:] / prices_np[:-1])
                pct_ret[1:] = prices_np[1:] / prices_np[:-1] - 1.0

        vola_s_x, vola_s_x2, vola_s_w, vola_s_w2, vola_count = _ewm_vol_accumulators_from_batch(
            log_ret, beta_vola, beta_vola_sq
        )
        pct_s_x, pct_s_x2, pct_s_w, pct_s_w2, pct_count = _ewm_vol_accumulators_from_batch(
            pct_ret, beta_vola, beta_vola_sq
        )

        # 5. Extract prev_cash_pos from WarmupState --------------------------
        prev_cash_pos: np.ndarray = ws.prev_cash_pos
        prev_price: np.ndarray = prices_np[-1].copy()

        # 6. Construct _StreamState and return ------------------------------
        state = _StreamState(
            corr_ret_buf=corr_ret_buf,
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
            prev_price=prev_price,
            prev_cash_pos=prev_cash_pos,
            step_count=n_rows,
            sw_ret_buf=sw_ret_buf,
        )
        return cls(cfg=cfg, assets=assets, state=state)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    @staticmethod
    def _warmup_threshold(cfg: BasanosConfig) -> int:
        """Return the step count at which warmup ends for the configured mode."""
        if isinstance(cfg.covariance_config, SlidingWindowConfig):
            return cfg.covariance_config.window
        return cfg.corr

    @staticmethod
    def _persist_state(
        state: _StreamState,
        *,
        corr_ret_buf: np.ndarray | None,
        vola_s_x: np.ndarray,
        vola_s_x2: np.ndarray,
        vola_s_w: np.ndarray,
        vola_s_w2: np.ndarray,
        vola_count: np.ndarray,
        pct_s_x: np.ndarray,
        pct_s_x2: np.ndarray,
        pct_s_w: np.ndarray,
        pct_s_w2: np.ndarray,
        pct_count: np.ndarray,
        new_price: np.ndarray,
        new_cash_pos: np.ndarray | None = None,
    ) -> None:
        """Persist accumulators, last-seen vectors, and increment step count."""
        state.corr_ret_buf = corr_ret_buf
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
        state.prev_price = new_price.copy()
        if new_cash_pos is not None:
            state.prev_cash_pos = new_cash_pos.copy()
        state.step_count += 1

    @staticmethod
    def _warmup_result(n_assets: int, date: Any) -> StepResult:
        """Build a standard warmup ``StepResult`` payload."""
        return StepResult(
            date=date,
            cash_position=np.full(n_assets, np.nan),
            status=SolveStatus.WARMUP,
            vola=np.full(n_assets, np.nan),
        )

    def _solve_sliding_window_position(
        self,
        *,
        cfg: BasanosConfig,
        state: _StreamState,
        mask: np.ndarray,
        new_m: np.ndarray,
        vola_vec: np.ndarray,
        n_assets: int,
        date: Any,
    ) -> tuple[np.ndarray, SolveStatus]:
        """Solve one step in SlidingWindow mode and return cash position + status."""
        from cvx.linalg import SingularMatrixError

        new_cash_pos = np.full(n_assets, np.nan, dtype=float)
        status = SolveStatus.DEGENERATE
        sw_config = cast(SlidingWindowConfig, cfg.covariance_config)
        if not mask.any():
            return new_cash_pos, status

        win_w = sw_config.window
        win_k = sw_config.n_factors
        window_ret = np.where(
            np.isfinite(state.sw_ret_buf[:, mask]),  # type: ignore[index]
            state.sw_ret_buf[:, mask],  # type: ignore[index]
            0.0,
        )
        n_sub = int(mask.sum())
        k_eff = min(win_k, win_w, n_sub)
        if sw_config.max_components is not None:
            k_eff = min(k_eff, sw_config.max_components)
        try:
            fm = FactorModel.from_returns(window_ret, k=k_eff)
        except (np.linalg.LinAlgError, ValueError) as exc:
            _logger.debug("Sliding window SVD failed at date=%s: %s", date, exc)
            new_cash_pos[mask] = 0.0
            return new_cash_pos, status

        expected_mu = np.nan_to_num(new_m[mask])
        if np.allclose(expected_mu, 0.0):
            new_cash_pos[mask] = 0.0
            return new_cash_pos, SolveStatus.ZERO_SIGNAL

        try:
            x = fm.solve(expected_mu)
            denom_val = float(np.sqrt(max(0.0, float(np.dot(expected_mu, x)))))
        except (SingularMatrixError, np.linalg.LinAlgError) as exc:
            _logger.warning("Woodbury solve failed at date=%s: %s", date, exc)
            new_cash_pos[mask] = 0.0
            return new_cash_pos, status

        if not np.isfinite(denom_val) or denom_val <= cfg.denom_tol:
            _logger.warning(
                "Positions zeroed at date=%s (sliding_window): normalisation "
                "denominator degenerate (denom=%s, denom_tol=%s).",
                date,
                denom_val,
                cfg.denom_tol,
            )
            new_cash_pos[mask] = 0.0
            return new_cash_pos, status

        risk_pos = x / denom_val
        vola_sub = vola_vec[mask]
        with np.errstate(invalid="ignore"):
            new_cash_pos[mask] = risk_pos / vola_sub
        return new_cash_pos, SolveStatus.VALID

    @staticmethod
    def _solve_ewma_position(
        *,
        cfg: BasanosConfig,
        state: _StreamState,
        corr_ret_buf: np.ndarray,
        mask: np.ndarray,
        new_m: np.ndarray,
        vola_vec: np.ndarray,
        assets: list[str],
        n_assets: int,
        date: Any,
    ) -> tuple[np.ndarray, SolveStatus]:
        """Solve one step in EWMA mode and return cash position + status."""
        new_cash_pos = np.full(n_assets, np.nan, dtype=float)
        buf = corr_ret_buf  # (T, N) — already includes the new row
        span = 2 * cfg.corr + 1
        t = buf.shape[0]
        cols = [pl.Series(a, buf[:, i]).fill_nan(None) for i, a in enumerate(assets)]
        pl_df = pl.DataFrame([pl.Series("t", list(range(t))), *cols])
        cov_dict = ewm_covariance(pl_df, assets=assets, index_col="t", window=span, warmup=cfg.corr)
        if not cov_dict:
            corr = np.full((n_assets, n_assets), np.nan)
        else:
            corr = cov_to_corr(cov_dict[max(cov_dict)], cfg.min_corr_denom)
        matrix = shrink2id(corr, lamb=cfg.shrink)
        expected_mu, early = _SolveMixin._row_early_check(state.step_count, date, mask, new_m)
        if early is not None:
            _, _, _, pos, status = early
            new_cash_pos[mask] = pos
            return new_cash_pos, status

        corr_sub = matrix[np.ix_(mask, mask)]
        _, _, _, pos, status = _SolveMixin._compute_position(
            state.step_count, date, mask, expected_mu, MatrixBundle(matrix=corr_sub), cfg.denom_tol
        )
        if status == SolveStatus.VALID:
            new_cash_pos[mask] = _SolveMixin._scale_to_cash(cast(np.ndarray, pos), vola_vec[mask])
        else:
            new_cash_pos[mask] = pos
        return new_cash_pos, status

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
            shape ``(N,)`` (assets ordered as in `assets`) or a dict
            mapping asset names to price values.
        new_mu:
            Per-asset expected-return signals, same format as ``new_prices``.
        date:
            Timestamp for this step (stored in `date`
            verbatim; not used in any computation).

        Returns:
        -------
        StepResult
            Frozen dataclass with ``cash_position``, ``vola``, ``status``, and
            ``date`` for this timestep.
        """
        cfg = self._cfg
        assets = self._assets
        state = self._state
        n_assets = len(assets)

        # ── Check if still in the warmup period ──────────────────────────────
        # step_count is initialised to n_rows in from_warmup.
        #
        # EwmaShrinkConfig: in_warmup is True for the first (cfg.corr - n_rows)
        # calls when the warmup batch was shorter than cfg.corr (not enough rows
        # to populate the EWM correlation matrix).
        #
        # SlidingWindowConfig: in_warmup is True for the first (window - n_rows)
        # calls when the warmup batch was shorter than the window.  During this
        # period sw_ret_buf still contains NaN-padded prefix rows; each step
        # shifts one NaN out and appends a real row, so the buffer is fully
        # populated with real data exactly when in_warmup becomes False.
        #
        # In both modes all accumulators are still updated during warmup so that
        # the state is ready the moment the warmup period ends.
        _warmup_thresh = self._warmup_threshold(cfg)
        in_warmup: bool = state.step_count < _warmup_thresh

        # ── Resolve inputs to (N,) float64 arrays ──────────────────────────
        new_p = _resolve_step_vector(new_prices, assets, n_assets, "new_prices")
        new_m = _resolve_step_vector(new_mu, assets, n_assets, "new_mu")

        prev_p = state.prev_price
        beta_vola: float = (cfg.vola - 1) / cfg.vola
        beta_vola_sq: float = beta_vola**2

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

        # ── Mode-specific correlation state update ───────────────────────────
        if isinstance(cfg.covariance_config, SlidingWindowConfig):
            # SW: shift the rolling window buffer in-place and append this row.
            buf = state.sw_ret_buf  # (W, N), already owned by state
            buf[:-1] = buf[1:]  # type: ignore[index]
            buf[-1] = vol_adj_val  # type: ignore[index]
            corr_ret_buf = state.corr_ret_buf  # None for SW; pass through
        else:
            # EWM: append new vol-adjusted return to the growing history buffer.
            new_row = vol_adj_val[np.newaxis]  # (1, N)
            corr_ret_buf = np.vstack([state.corr_ret_buf, new_row])

        # ── Early return during EWM warmup period ───────────────────────────
        # All accumulators are already updated above; skip the O(N²) matrix
        # reconstruction and O(N³) Cholesky solve which are wasteful during
        # warmup — the computed positions would be discarded anyway.
        if in_warmup:
            self._persist_state(
                state,
                corr_ret_buf=corr_ret_buf,
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
                new_price=new_p,
            )
            return self._warmup_result(n_assets, date)

        # ── Compute EWMA volatility (pct-return std) — shared ───────────────
        vola_vec = _ewm_std_from_state(pct_s_x, pct_s_x2, pct_s_w, pct_s_w2, pct_count, min_samples=cfg.vola)

        # ── Solve for position ───────────────────────────────────────────────
        mask = np.isfinite(new_p)
        if isinstance(cfg.covariance_config, SlidingWindowConfig):
            new_cash_pos, status = self._solve_sliding_window_position(
                cfg=cfg,
                state=state,
                mask=mask,
                new_m=new_m,
                vola_vec=vola_vec,
                n_assets=n_assets,
                date=date,
            )
        else:
            new_cash_pos, status = self._solve_ewma_position(
                cfg=cfg,
                state=state,
                corr_ret_buf=corr_ret_buf,
                mask=mask,
                new_m=new_m,
                vola_vec=vola_vec,
                assets=list(self._assets),
                n_assets=n_assets,
                date=date,
            )

        # ── Apply turnover constraint ─────────────────────────────────────────
        if cfg.max_turnover is not None and status == SolveStatus.VALID:
            new_cash_pos[mask] = _SolveMixin._apply_turnover_constraint(
                new_cash_pos[mask],
                state.prev_cash_pos[mask],
                cfg.max_turnover,
            )

        # ── Persist updated state ───────────────────────────────────────────
        self._persist_state(
            state,
            corr_ret_buf=corr_ret_buf,
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
            new_price=new_p,
            new_cash_pos=new_cash_pos,
        )

        return StepResult(
            date=date,
            cash_position=new_cash_pos,
            status=status,
            vola=vola_vec,
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        """Serialise the stream to a ``.npz`` archive at *path*.

        All `_StreamState` arrays, the configuration, and the asset
        list are written in a single `savez` call.  A stream
        restored via `load` produces bit-for-bit identical
        `step` output.

        Args:
            path: Destination file path.  `savez` appends
                ``.npz`` automatically when the suffix is absent.

        Examples:
            >>> import tempfile, pathlib, numpy as np
            >>> import polars as pl
            >>> from datetime import date, timedelta
            >>> from basanos.math import BasanosConfig, BasanosStream
            >>> rng = np.random.default_rng(0)
            >>> n = 60
            >>> end = date(2024, 1, 1) + timedelta(days=n - 1)
            >>> dates = pl.date_range(
            ...     date(2024, 1, 1), end, interval="1d", eager=True
            ... )
            >>> prices = pl.DataFrame({
            ...     "date": dates,
            ...     "A": np.cumprod(1 + rng.normal(0.001, 0.02, n)) * 100.0,
            ...     "B": np.cumprod(1 + rng.normal(0.001, 0.02, n)) * 150.0,
            ... })
            >>> mu = pl.DataFrame({
            ...     "date": dates,
            ...     "A": rng.normal(0, 0.5, n),
            ...     "B": rng.normal(0, 0.5, n),
            ... })
            >>> cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)
            >>> stream = BasanosStream.from_warmup(prices, mu, cfg)
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     p = pathlib.Path(tmp) / "stream.npz"
            ...     stream.save(p)
            ...     restored = BasanosStream.load(p)
            ...     restored.assets == stream.assets
            True
        """
        state = self._state
        # Build the per-field dict automatically from _StreamState so that any
        # new field added to the dataclass is included without manual updates.
        state_arrays: dict[str, Any] = {}
        for field in dataclasses.fields(_StreamState):
            value = getattr(state, field.name)
            if field.name in ("sw_ret_buf", "corr_ret_buf"):
                # Sentinel: use an empty (0, 0) array to represent None so the
                # key is always present in the archive and load() can detect it.
                state_arrays[field.name] = value if value is not None else np.empty((0, 0), dtype=float)
            elif field.name == "step_count":
                state_arrays[field.name] = np.array(value)
            else:
                state_arrays[field.name] = value
        np.savez(
            path,
            format_version=np.array(_SAVE_FORMAT_VERSION),
            cfg_json=np.array(self._cfg.model_dump_json()),
            assets=np.array(self._assets),
            **state_arrays,
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> BasanosStream:
        """Restore a stream previously saved with `save`.

        Args:
            path: Path to a ``.npz`` archive written by `save`.

        Returns:
            A `BasanosStream` whose `step` output is
            bit-for-bit identical to the original stream at the time
            `save` was called.

        Examples:
            >>> import tempfile, pathlib, numpy as np
            >>> import polars as pl
            >>> from datetime import date, timedelta
            >>> from basanos.math import BasanosConfig, BasanosStream
            >>> rng = np.random.default_rng(1)
            >>> n = 60
            >>> end = date(2024, 1, 1) + timedelta(days=n - 1)
            >>> dates = pl.date_range(
            ...     date(2024, 1, 1), end, interval="1d", eager=True
            ... )
            >>> prices = pl.DataFrame({
            ...     "date": dates,
            ...     "A": np.cumprod(1 + rng.normal(0.001, 0.02, n)) * 100.0,
            ...     "B": np.cumprod(1 + rng.normal(0.001, 0.02, n)) * 150.0,
            ... })
            >>> mu = pl.DataFrame({
            ...     "date": dates,
            ...     "A": rng.normal(0, 0.5, n),
            ...     "B": rng.normal(0, 0.5, n),
            ... })
            >>> cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)
            >>> stream = BasanosStream.from_warmup(prices, mu, cfg)
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     p = pathlib.Path(tmp) / "stream.npz"
            ...     stream.save(p)
            ...     restored = BasanosStream.load(p)
            ...     restored.assets == stream.assets
            True
        """
        with np.load(path, allow_pickle=False) as data:
            if "format_version" not in data:
                raise ValueError(  # noqa: TRY003
                    "Stream file is missing a format version tag. "
                    "It was written with an incompatible version of BasanosStream. "
                    "Re-generate it via BasanosStream.from_warmup()."
                )
            found = int(data["format_version"])
            if found != _SAVE_FORMAT_VERSION:
                raise ValueError(  # noqa: TRY003
                    f"Stream file was written with format version {found}, "
                    f"but the current version is {_SAVE_FORMAT_VERSION}. "
                    "Re-generate it via BasanosStream.from_warmup()."
                )
            # Validate that every required key is present.  This catches archives
            # that were produced by an older codebase missing a newly added field,
            # or archives that have been manually edited, with a descriptive error
            # instead of a bare KeyError.
            archive_keys = frozenset(data.files)
            missing = _REQUIRED_KEYS - archive_keys
            if missing:
                raise StreamStateCorruptError(missing)
            cfg = BasanosConfig.model_validate_json(data["cfg_json"].item())
            assets: list[str] = list(data["assets"])
            state_kwargs: dict[str, Any] = {}
            for field in dataclasses.fields(_StreamState):
                raw = data[field.name]
                if field.name in ("sw_ret_buf", "corr_ret_buf"):
                    state_kwargs[field.name] = raw if raw.size > 0 else None
                elif field.name == "step_count":
                    state_kwargs[field.name] = int(raw)
                else:
                    state_kwargs[field.name] = raw
        state = _StreamState(**state_kwargs)
        return cls(cfg=cfg, assets=assets, state=state)
