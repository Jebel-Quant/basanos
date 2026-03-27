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
import os
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ._ewm_corr import _EwmCorrState

import numpy as np
import polars as pl
from scipy.signal import lfilter

from ..exceptions import MissingDateColumnError, StreamStateCorruptError
from ._config import BasanosConfig, EwmaShrinkConfig, SlidingWindowConfig
from ._engine_solve import MatrixBundle, SolveStatus, _SolveMixin
from ._ewm_corr import _corr_from_ewm_accumulators
from ._factor_model import FactorModel
from ._signal import shrink2id

_logger = logging.getLogger(__name__)

#: Increment this when the :func:`BasanosStream.save` archive layout changes in
#: a backward-incompatible way.  :func:`BasanosStream.load` asserts the stored
#: value matches before deserialising anything, so callers get a clear error
#: instead of a silent ``KeyError`` or wrong state.
_SAVE_FORMAT_VERSION: int = 2


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
        prev_price:    Last price row seen, used to compute returns on the next
            step; shape ``(N,)``.
        prev_cash_pos: Last cash position, used to apply the turnover constraint
            on the next step; shape ``(N,)``.
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
    prev_price: np.ndarray  # (N,) last price row (to compute returns at next step)
    prev_cash_pos: np.ndarray  # (N,) last cash position (for turnover constraint at next step)
    step_count: int

    # ── SlidingWindowConfig state — None for EwmaShrinkConfig ────────────────
    # shape (W, N): last W vol-adjusted returns (oldest row first); None when
    # using EwmaShrinkConfig.  The corr_zi_* fields above are unused (zeros)
    # in this mode; sw_ret_buf carries all the correlation state instead.
    sw_ret_buf: np.ndarray | None = None  # (W, N) rolling buffer, or None


#: Keys that :meth:`BasanosStream.save` writes to the ``.npz`` archive for
#: :class:`_StreamState` fields.  Derived automatically from
#: :func:`dataclasses.fields` so that adding a new field to ``_StreamState``
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
            DataFrame (typically a Python :class:`datetime.date`,
            :class:`datetime.datetime`, or a Polars temporal scalar).
        cash_position: Optimised cash-position vector, shape ``(N,)``.
            Entries are ``NaN`` for assets that are still in the EWMA warmup
            period or that are otherwise inactive at this step.
        status: Solver outcome label for this timestep
            (:class:`~basanos.math.SolveStatus`).  Since :class:`SolveStatus`
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

    Implements the same IIR recurrence as :meth:`BasanosStream.step` but
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
    :meth:`BasanosStream.from_warmup` for both the log-return (``vola_*``)
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
            Engine configuration.  Both :class:`~basanos.math.EwmaShrinkConfig`
            and :class:`~basanos.math.SlidingWindowConfig` are supported.

        Returns:
        -------
        BasanosStream
            A stream instance whose :meth:`step` method is ready to accept the
            row immediately following the last warmup row.

        Notes:
        ------
        **Short-warmup behaviour with** ``SlidingWindowConfig``: when
        ``len(prices) < cfg.covariance_config.window``, the internal rolling
        buffer (``sw_ret_buf``) is NaN-padded for the missing prefix rows.
        :meth:`step` returns ``StepResult(status="warmup")`` for each of the
        first ``window - len(prices)`` calls, exactly matching the EWM warmup
        semantics.  By the time :meth:`step` returns the first non-warmup
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
            # EWM: seed the per-step lfilter from the IIR state captured
            # during the single batch pass in warmup_state().
            iir = cast("_EwmCorrState", ws.corr_iir_state)
            corr_zi_x = iir.corr_zi_x
            corr_zi_x2 = iir.corr_zi_x2
            corr_zi_xy = iir.corr_zi_xy
            corr_zi_w = iir.corr_zi_w
            corr_count: np.ndarray = iir.count
            sw_ret_buf: np.ndarray | None = None
        else:
            # SW: carry the last W vol-adjusted returns as a rolling buffer.
            # The IIR fields are initialised to zeros and left unused.
            sw_config = cast(SlidingWindowConfig, cfg.covariance_config)
            win_w = sw_config.window
            ret_adj_np = engine.ret_adj.select(assets).to_numpy()  # (n_rows, N)
            if n_rows >= win_w:
                sw_ret_buf = ret_adj_np[-win_w:].copy()
            else:
                sw_ret_buf = np.full((win_w, n_assets), np.nan)
                sw_ret_buf[-n_rows:] = ret_adj_np
            corr_zi_x = np.zeros((1, n_assets, n_assets))
            corr_zi_x2 = np.zeros((1, n_assets, n_assets))
            corr_zi_xy = np.zeros((1, n_assets, n_assets))
            corr_zi_w = np.zeros((1, n_assets, n_assets))
            corr_count = np.zeros((n_assets, n_assets), dtype=np.int64)

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
            prev_price=prev_price,
            prev_cash_pos=prev_cash_pos,
            step_count=n_rows,
            sw_ret_buf=sw_ret_buf,
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
        _warmup_thresh = (
            cfg.covariance_config.window if isinstance(cfg.covariance_config, SlidingWindowConfig) else cfg.corr
        )
        in_warmup: bool = state.step_count < _warmup_thresh

        # ── Resolve inputs to (N,) float64 arrays ──────────────────────────
        if isinstance(new_prices, dict):
            new_p = np.array([float(new_prices[a]) for a in assets], dtype=float)
        else:
            new_p = np.asarray(new_prices, dtype=float).ravel()

        if isinstance(new_mu, dict):
            new_m = np.array([float(new_mu[a]) for a in assets], dtype=float)
        else:
            new_m = np.asarray(new_mu, dtype=float).ravel()

        if new_p.shape != (n_assets,):
            raise ValueError(f"new_prices must have shape ({n_assets},); got {new_p.shape}")  # noqa: TRY003
        if new_m.shape != (n_assets,):
            raise ValueError(f"new_mu must have shape ({n_assets},); got {new_m.shape}")  # noqa: TRY003

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

        # ── Mode-specific correlation state update ───────────────────────────
        if isinstance(cfg.covariance_config, SlidingWindowConfig):
            # SW: shift the rolling window buffer in-place and append this row.
            # The corr_zi_* fields are unused; alias them to their old values so
            # the early-return and persist blocks below can reference them safely.
            buf = state.sw_ret_buf  # (W, N), already owned by state
            buf[:-1] = buf[1:]  # type: ignore[index]
            buf[-1] = vol_adj_val  # type: ignore[index]
            corr_zi_x = state.corr_zi_x
            corr_zi_x2 = state.corr_zi_x2
            corr_zi_xy = state.corr_zi_xy
            corr_zi_w = state.corr_zi_w
            corr_count = state.corr_count
        else:
            # EWM: Update IIR filter state for EWM correlation
            fin_va = np.isfinite(vol_adj_val)
            va_f = np.where(fin_va, vol_adj_val, 0.0)
            joint_fin = fin_va[:, np.newaxis] & fin_va[np.newaxis, :]  # (N, N)

            new_v_x = (va_f[:, np.newaxis] * joint_fin)[np.newaxis]  # (1, N, N)
            new_v_x2 = ((va_f**2)[:, np.newaxis] * joint_fin)[np.newaxis]  # (1, N, N)
            new_v_xy = (va_f[:, np.newaxis] * va_f[np.newaxis, :])[np.newaxis]  # (1, N, N)
            new_v_w = joint_fin.astype(np.float64)[np.newaxis]  # (1, N, N)

            filt_a_corr = np.array([1.0, -beta_corr])
            # y_x[0] is the current-step EWM state (filter output); corr_zi_x is
            # the new filter memory (zf = beta * y[0]) passed as zi next step.
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
                status=SolveStatus.WARMUP,
                vola=np.full(n_assets, np.nan),
            )

        # ── Compute EWMA volatility (pct-return std) — shared ───────────────
        vola_vec = _ewm_std_from_state(pct_s_x, pct_s_x2, pct_s_w, pct_s_w2, pct_count, min_samples=cfg.vola)

        # ── Solve for position ───────────────────────────────────────────────
        new_cash_pos = np.full(n_assets, np.nan, dtype=float)
        status = SolveStatus.DEGENERATE

        mask = np.isfinite(new_p)

        if isinstance(cfg.covariance_config, SlidingWindowConfig):
            # ── SW path: FactorModel solve via Woodbury identity ─────────────
            sw_config = cfg.covariance_config
            if not mask.any():
                status = SolveStatus.DEGENERATE
            else:
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
                    status = SolveStatus.DEGENERATE
                else:
                    expected_mu = np.nan_to_num(new_m[mask])
                    if np.allclose(expected_mu, 0.0):
                        new_cash_pos[mask] = 0.0
                        status = SolveStatus.ZERO_SIGNAL
                    else:
                        try:
                            x = fm.solve(expected_mu)
                            denom_val = float(np.sqrt(max(0.0, float(np.dot(expected_mu, x)))))
                        except (SingularMatrixError, np.linalg.LinAlgError) as exc:
                            _logger.warning("Woodbury solve failed at date=%s: %s", date, exc)
                            new_cash_pos[mask] = 0.0
                            status = SolveStatus.DEGENERATE
                        else:
                            if not np.isfinite(denom_val) or denom_val <= cfg.denom_tol:
                                _logger.warning(
                                    "Positions zeroed at date=%s (sliding_window): normalisation "
                                    "denominator degenerate (denom=%s, denom_tol=%s).",
                                    date,
                                    denom_val,
                                    cfg.denom_tol,
                                )
                                new_cash_pos[mask] = 0.0
                                status = SolveStatus.DEGENERATE
                            else:
                                risk_pos = x / denom_val
                                vola_sub = vola_vec[mask]
                                with np.errstate(invalid="ignore"):
                                    new_cash_pos[mask] = risk_pos / vola_sub
                                status = SolveStatus.VALID
        else:
            # ── EWM path: shared _corr_from_ewm_accumulators + _SolveMixin solve ──
            # Reconstruct the EWM correlation matrix from filter outputs using
            # the shared helper — the same formula as _ewm_corr_with_final_state
            # but operating on (N, N) slices instead of (T, N, N) tensors.
            # Use y_*[0] (the filter OUTPUT for this step), not zf[0].
            corr = _corr_from_ewm_accumulators(
                y_x[0],
                y_x2[0],
                y_xy[0],
                y_w[0],
                corr_count,
                min_periods=cfg.corr,
                min_corr_denom=cfg.min_corr_denom,
            )
            matrix = shrink2id(corr, lamb=cfg.shrink)

            # Delegate the signal check and linear solve to _SolveMixin so that
            # any algorithm change (denominator guard, status labels, etc.) only
            # needs to be applied in _engine_solve.py.
            expected_mu, early = _SolveMixin._row_early_check(state.step_count, date, mask, new_m)
            if early is not None:
                _, _, _, pos, status = early
                new_cash_pos[mask] = pos
            else:
                corr_sub = matrix[np.ix_(mask, mask)]
                _, _, _, pos, status = _SolveMixin._compute_position(
                    state.step_count, date, mask, expected_mu, MatrixBundle(matrix=corr_sub), cfg.denom_tol
                )
                if status == SolveStatus.VALID:
                    new_cash_pos[mask] = _SolveMixin._scale_to_cash(cast(np.ndarray, pos), vola_vec[mask])
                else:
                    new_cash_pos[mask] = pos

        # ── Apply turnover constraint ─────────────────────────────────────────
        if cfg.max_turnover is not None and status == SolveStatus.VALID:
            new_cash_pos[mask] = _SolveMixin._apply_turnover_constraint(
                new_cash_pos[mask],
                state.prev_cash_pos[mask],
                cfg.max_turnover,
            )

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
        state.step_count += 1

        return StepResult(
            date=date,
            cash_position=new_cash_pos,
            status=status,
            vola=vola_vec,
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        """Serialise the stream to a ``.npz`` archive at *path*.

        All :class:`_StreamState` arrays, the configuration, and the asset
        list are written in a single :func:`numpy.savez` call.  A stream
        restored via :meth:`load` produces bit-for-bit identical
        :meth:`step` output.

        Args:
            path: Destination file path.  :func:`numpy.savez` appends
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
            if field.name == "sw_ret_buf":
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
        """Restore a stream previously saved with :meth:`save`.

        Args:
            path: Path to a ``.npz`` archive written by :meth:`save`.

        Returns:
            A :class:`BasanosStream` whose :meth:`step` output is
            bit-for-bit identical to the original stream at the time
            :meth:`save` was called.

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
        data = np.load(path, allow_pickle=False)
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
            if field.name == "sw_ret_buf":
                state_kwargs[field.name] = raw if raw.size > 0 else None
            elif field.name == "step_count":
                state_kwargs[field.name] = int(raw)
            else:
                state_kwargs[field.name] = raw
        state = _StreamState(**state_kwargs)
        return cls(cfg=cfg, assets=assets, state=state)
