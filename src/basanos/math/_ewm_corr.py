"""Pure-NumPy exponentially weighted moving correlation.

This private module contains :func:`_ewm_corr_numpy`, the vectorised
IIR-filter implementation of per-row EWM correlation matrices.  It is
separated from :mod:`basanos.math.optimizer` so that the algorithm can
be read, tested, and profiled in isolation without loading the full
engine machinery.
"""

import numpy as np
from scipy.signal import lfilter


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
