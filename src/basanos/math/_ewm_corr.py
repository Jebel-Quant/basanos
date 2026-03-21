"""Pure-NumPy exponentially weighted moving correlation.

This private module contains :func:`_ewm_corr_numpy`, the vectorised
IIR-filter implementation of per-row EWM correlation matrices.  It is
separated from :mod:`basanos.math.optimizer` so that the algorithm can
be read, tested, and profiled in isolation without loading the full
engine machinery.
"""

import dataclasses

import numpy as np
from scipy.signal import lfilter


@dataclasses.dataclass(frozen=True)
class CorrIirState:
    """Final IIR filter states from the EWM correlation computation.

    Captures the four ``zf`` arrays produced by :func:`_ewm_corr_iir_zf`
    (or the ``return_iir_state=True`` path of :func:`_ewm_corr_numpy`) so
    that a :class:`~basanos.math.BasanosStream` can resume IIR filtering at
    the next timestep without replaying the full warmup batch.

    Attributes:
        zi_x: Final state for the ``s_x`` accumulator, shape ``(1, N, N)``.
        zi_x2: Final state for the ``s_x2`` accumulator, shape ``(1, N, N)``.
        zi_xy: Final state for the ``s_xy`` accumulator, shape ``(1, N, N)``.
        zi_w: Final state for the ``s_w`` (weight) accumulator,
            shape ``(1, N, N)``.
        count: Cumulative joint-finite observation count per asset pair,
            shape ``(N, N)``, dtype ``int64``.
    """

    zi_x: np.ndarray
    zi_x2: np.ndarray
    zi_xy: np.ndarray
    zi_w: np.ndarray
    count: np.ndarray


def _ewm_corr_numpy(
    data: np.ndarray,
    com: int,
    min_periods: int,
    min_corr_denom: float = 1e-14,
) -> np.ndarray:
    """Compute per-row EWM correlation matrices without pandas.

    Matches ``pandas.DataFrame.ewm(com=com, min_periods=min_periods).corr()``
    with the default ``adjust=True, ignore_na=False`` settings to within
    floating-point rounding error.

    All five EWM components used to compute ``corr(i, j)`` — namely
    ``ewm(x_i)``, ``ewm(x_j)``, ``ewm(x_i²)``, ``ewm(x_j²)``, and
    ``ewm(x_i·x_j)`` — share the **same joint weight structure**: weights
    decay at every timestep (``ignore_na=False``) but a new observation is
    only added at timesteps where *both* ``x_i`` and ``x_j`` are finite.  As
    a result the correlation for a pair is frozen once either asset goes
    missing, exactly mirroring pandas behaviour.

    The EWM recurrence ``s[t] = β·s[t-1] + v[t]`` is an IIR filter and is
    solved for **all N² pairs simultaneously** via ``scipy.signal.lfilter``
    — no Python loop over the T timesteps.

    Args:
        data: Float array of shape ``(T, N)`` - typically volatility-adjusted
            log returns.
        com: EWM centre-of-mass (``alpha = 1 / (1 + com)``).
        min_periods: Minimum number of joint finite observations required
            before a correlation value is reported; earlier rows are NaN.
        min_corr_denom: Guard threshold below which the correlation denominator
            is treated as zero and the result is set to NaN.  Defaults to
            ``1e-14``.

    Returns:
        np.ndarray of shape ``(T, N, N)`` containing the per-row correlation
        matrices.  Each matrix is symmetric with diagonal 1.0 (or NaN during
        warm-up).

    Performance:
        **Time** — O(T·N²): ``lfilter`` processes all N² pairs simultaneously,
        so wall-clock time scales linearly with both T and N².

        **Memory** — approximately 14 float64 arrays of shape ``(T, N, N)``
        exist at peak, giving roughly ``112 * T * N²`` bytes.  For 100 assets
        over 2 520 trading days (~10 years) that is ≈ 2.8 GB; for 500 assets
        the same period requires ≈ 70 GB, which exceeds typical workstation
        RAM.  Reduce T or N before calling this function when working with
        large universes.
    """
    _t_len, n_assets = data.shape
    beta = com / (1.0 + com)

    fin = np.isfinite(data)  # (T, N) bool
    xt_f = np.where(fin, data, 0.0)  # (T, N) float - zeroed where not finite

    # joint_fin[t, i, j] = True iff assets i and j are both finite at t
    joint_fin = fin[:, :, np.newaxis] & fin[:, np.newaxis, :]  # (T, N, N)

    # Build per-pair input sequences for the recurrence s[t] = beta*s[t-1] + v[t].
    #
    # v_x[t, i, j]  = x_i[t]    where pair (i,j) jointly finite, else 0
    # v_x2[t, i, j] = x_i[t]^2  where jointly finite, else 0
    # v_xy[t, i, j] = x_i[t]*x_j[t]  (xt_f is 0 for non-finite, so implicit mask)
    # v_w[t, i, j]  = 1          where jointly finite, else 0  (weight indicator)
    #
    # By symmetry v_x[t,j,i] carries x_j[t] for pair (i,j), so s_x.swapaxes(1,2)
    # gives the EWM numerator of x_j without a separate v_y array.
    v_x = xt_f[:, :, np.newaxis] * joint_fin  # (T, N, N)
    v_x2 = (xt_f * xt_f)[:, :, np.newaxis] * joint_fin  # (T, N, N)
    v_xy = xt_f[:, :, np.newaxis] * xt_f[:, np.newaxis, :]  # (T, N, N)
    v_w = joint_fin.astype(np.float64)  # (T, N, N)

    # Solve the IIR recurrence for every (i, j) pair in parallel.
    # lfilter([1], [1, -beta], v, axis=0) computes s[t] = beta*s[t-1] + v[t].
    filt_a = np.array([1.0, -beta])
    s_x = lfilter([1.0], filt_a, v_x, axis=0)  # (T, N, N)
    s_x2 = lfilter([1.0], filt_a, v_x2, axis=0)  # (T, N, N)
    s_xy = lfilter([1.0], filt_a, v_xy, axis=0)  # (T, N, N)
    s_w = lfilter([1.0], filt_a, v_w, axis=0)  # (T, N, N)

    # Joint finite observation count per pair at each timestep (for min_periods)
    count = np.cumsum(joint_fin, axis=0)  # (T, N, N) int64

    # EWM means: running numerator / running weight denominator.
    # s_x.swapaxes(1,2)[t,i,j] = s_x[t,j,i] = EWM numerator of x_j for pair (i,j).
    with np.errstate(divide="ignore", invalid="ignore"):
        pos_w = s_w > 0
        ewm_x = np.where(pos_w, s_x / s_w, np.nan)  # EWM(x_i)
        ewm_y = np.where(pos_w, s_x.swapaxes(1, 2) / s_w, np.nan)  # EWM(x_j)
        ewm_x2 = np.where(pos_w, s_x2 / s_w, np.nan)  # EWM(x_i^2)
        ewm_y2 = np.where(pos_w, s_x2.swapaxes(1, 2) / s_w, np.nan)  # EWM(x_j^2)
        ewm_xy = np.where(pos_w, s_xy / s_w, np.nan)  # EWM(x_i*x_j)

    var_x = np.maximum(ewm_x2 - ewm_x * ewm_x, 0.0)
    var_y = np.maximum(ewm_y2 - ewm_y * ewm_y, 0.0)
    denom = np.sqrt(var_x * var_y)
    cov = ewm_xy - ewm_x * ewm_y

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom > min_corr_denom, cov / denom, np.nan)

    result = np.clip(result, -1.0, 1.0)

    # Apply min_periods mask for all pairs
    result[count < min_periods] = np.nan

    # Diagonal is exactly 1.0 where the asset has sufficient observations
    diag_idx = np.arange(n_assets)
    diag_count = count[:, diag_idx, diag_idx]  # (T, N)
    result[:, diag_idx, diag_idx] = np.where(diag_count >= min_periods, 1.0, np.nan)

    return result


def _ewm_corr_iir_zf(data: np.ndarray, com: int) -> CorrIirState:
    """Return the final IIR filter states for the EWM correlation computation.

    Runs the same four ``lfilter`` sweeps as :func:`_ewm_corr_numpy` but
    discards the intermediate per-timestep outputs and returns only the
    terminal ``zf`` arrays together with the cumulative joint-finite count.
    This is used by :meth:`~basanos.math.BasanosEngine.warmup_state` to
    populate :class:`~basanos.math._engine_solve.WarmupState` so that
    :meth:`~basanos.math.BasanosStream.from_warmup` can resume streaming
    without a second pass over the warmup data.

    Args:
        data: Float array of shape ``(T, N)`` — typically volatility-adjusted
            log returns, matching the input to :func:`_ewm_corr_numpy`.
        com: EWM centre-of-mass (``alpha = 1 / (1 + com)``).

    Returns:
        :class:`CorrIirState` with ``zi_x``, ``zi_x2``, ``zi_xy``, ``zi_w``
        each of shape ``(1, N, N)`` and ``count`` of shape ``(N, N)``.
    """
    _t_len, n_assets = data.shape
    beta = com / (1.0 + com)

    fin = np.isfinite(data)
    xt_f = np.where(fin, data, 0.0)
    joint_fin = fin[:, :, np.newaxis] & fin[:, np.newaxis, :]  # (T, N, N)

    v_x = xt_f[:, :, np.newaxis] * joint_fin  # (T, N, N)
    v_x2 = (xt_f * xt_f)[:, :, np.newaxis] * joint_fin  # (T, N, N)
    v_xy = xt_f[:, :, np.newaxis] * xt_f[:, np.newaxis, :]  # (T, N, N)
    v_w = joint_fin.astype(np.float64)  # (T, N, N)

    filt_a = np.array([1.0, -beta])
    zi0 = np.zeros((1, n_assets, n_assets))
    _, zf_x = lfilter([1.0], filt_a, v_x, axis=0, zi=zi0)
    _, zf_x2 = lfilter([1.0], filt_a, v_x2, axis=0, zi=zi0)
    _, zf_xy = lfilter([1.0], filt_a, v_xy, axis=0, zi=zi0)
    _, zf_w = lfilter([1.0], filt_a, v_w, axis=0, zi=zi0)

    count = np.sum(joint_fin.astype(np.int64), axis=0)  # (N, N)

    return CorrIirState(zi_x=zf_x, zi_x2=zf_x2, zi_xy=zf_xy, zi_w=zf_w, count=count)
