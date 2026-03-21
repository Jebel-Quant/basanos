"""Mutable state carrier for the incremental :class:`BasanosStream` API.

This private module defines :class:`_StreamState`, a plain (non-frozen)
dataclass that persists all O(N²) state between consecutive
``BasanosStream.step()`` calls.  It is intentionally kept separate from the
engine so the state layout can be read and tested in isolation.

IIR state model
---------------
The EWM recurrence ``s[t] = β·s[t-1] + v[t]`` is a causal, single-pole IIR
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
giving **O(N²)** memory independent of the number of timesteps processed.
"""

import dataclasses

import numpy as np


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
