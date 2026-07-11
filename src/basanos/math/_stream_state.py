"""State containers for the incremental (streaming) optimiser.

Defines the mutable per-instance state carrier (`_StreamState`), the frozen
per-step output (`StepResult`), and the on-disk format version that the
persistence layer in :mod:`basanos.math._stream_io` writes and validates.

Keeping these types in their own module lets the state layout be read and
tested in isolation, and lets the persistence and solver layers depend on the
state representation without importing the `BasanosStream` façade.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ._engine_solve import SolveStatus

#: Increment this when the ``save`` archive layout changes in
#: a backward-incompatible way.  ``load`` asserts the stored
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

    def persist(
        self,
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
        self.corr_ret_buf = corr_ret_buf
        self.vola_s_x = vola_s_x
        self.vola_s_x2 = vola_s_x2
        self.vola_s_w = vola_s_w
        self.vola_s_w2 = vola_s_w2
        self.vola_count = vola_count
        self.pct_s_x = pct_s_x
        self.pct_s_x2 = pct_s_x2
        self.pct_s_w = pct_s_w
        self.pct_s_w2 = pct_s_w2
        self.pct_count = pct_count
        self.prev_price = new_price.copy()
        if new_cash_pos is not None:
            self.prev_cash_pos = new_cash_pos.copy()
        self.step_count += 1


#: Keys that ``save`` writes to the ``.npz`` archive for
#: `_StreamState` fields.  Derived automatically from
#: `dataclasses.fields` so that adding a new field to ``_StreamState``
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
