"""Pure-NumPy exponentially weighted moving correlation.

This module contains `ewm_corr`, the vectorised
IIR-filter implementation of per-row EWM correlation matrices.  It is
separated from `optimizer` so that the algorithm can
be read, tested, and profiled in isolation without loading the full
engine machinery.
"""

import dataclasses

import numpy as np
from scipy.signal import lfilter


@dataclasses.dataclass
class _EwmCorrState:
    """Final IIR filter memory after a full EWM correlation pass.

    Returned alongside the correlation tensor by
    `_ewm_corr_with_final_state`.  Pass these values as the ``zi``
    arguments to ``scipy.signal.lfilter`` on the next step to continue the
    IIR recurrence without replaying history.

    For a first-order filter ``a = [1, -beta]``, ``b = [1]``, the final
    state after processing the last sample ``y[-1]`` is ``zf[0] = beta *
    y[-1]``.  Each ``corr_zi_*`` field stores exactly that quantity,
    shaped ``(1, N, N)`` to match the ``zi`` format expected by
    ``lfilter(..., axis=0)``.

    Attributes:
        corr_zi_x:  Final state for the ``x``-accumulator; shape ``(1, N, N)``.
        corr_zi_x2: Final state for the ``x²``-accumulator; shape ``(1, N, N)``.
        corr_zi_xy: Final state for the ``xy``-accumulator; shape ``(1, N, N)``.
        corr_zi_w:  Final state for the weight-accumulator; shape ``(1, N, N)``.
        count:      Cumulative joint-finite observation count per asset pair
                    at the last timestep; shape ``(N, N)`` dtype int64.
    """

    corr_zi_x: np.ndarray  # (1, N, N)
    corr_zi_x2: np.ndarray  # (1, N, N)
    corr_zi_xy: np.ndarray  # (1, N, N)
    corr_zi_w: np.ndarray  # (1, N, N)
    count: np.ndarray  # (N, N) int64


def _corr_from_ewm_accumulators(
    s_x: np.ndarray,
    s_x2: np.ndarray,
    s_xy: np.ndarray,
    s_w: np.ndarray,
    count: np.ndarray,
    min_periods: int,
    min_corr_denom: float = 1e-14,
) -> np.ndarray:
    """Compute a single EWM correlation matrix from running accumulators.

    Single-timestep equivalent of `_ewm_corr_with_final_state`: applies
    the same formula to ``(N, N)`` arrays instead of ``(T, N, N)`` tensors.
    Called by `step` after advancing the IIR filter state
    by one row to reconstruct the current correlation matrix without revisiting
    history.

    This is the **canonical** implementation of the EWM correlation formula
    shared by both the batch and the incremental paths.  Any change to the
    formula (e.g. symmetrisation, denominator guard, NaN-masking) must be made
    here only — `_ewm_corr_with_final_state` delegates its inner
    computation to this function so that a single definition is maintained.

    Args:
        s_x:  Running EWM sum of x; shape ``(N, N)``.
        s_x2: Running EWM sum of x²; shape ``(N, N)``.
        s_xy: Running EWM sum of x·y; shape ``(N, N)``.
        s_w:  Running EWM sum of weights; shape ``(N, N)``.
        count: Cumulative joint-finite observation count; shape ``(N, N)``
            dtype int.
        min_periods: Minimum joint-finite observations before a value is
            reported; earlier entries are ``NaN``.
        min_corr_denom: Guard threshold below which the correlation denominator
            is treated as zero and the result is set to ``NaN``.

    Returns:
        np.ndarray of shape ``(N, N)``: symmetric correlation matrix with
        diagonal ``1.0`` (or ``NaN`` during warm-up) and off-diagonal entries
        in ``[-1, 1]``.
    """
    n_assets = s_x.shape[0]

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
        corr = np.where(denom_corr > min_corr_denom, cov / denom_corr, np.nan)
    corr = np.clip(corr, -1.0, 1.0)
    corr[count < min_periods] = np.nan

    diag_idx = np.arange(n_assets)
    corr[diag_idx, diag_idx] = np.where(count[diag_idx, diag_idx] >= min_periods, 1.0, np.nan)

    # Enforce symmetry: average strict lower/upper triangle pairs in place.
    tril_i, tril_j = np.tril_indices(n_assets, k=-1)
    upper_vals = corr[tril_j, tril_i]
    lower_vals = corr[tril_i, tril_j]
    avg_vals = 0.5 * (upper_vals + lower_vals)
    corr[tril_i, tril_j] = avg_vals
    corr[tril_j, tril_i] = avg_vals

    return corr


def _ewm_corr_with_final_state(
    data: np.ndarray,
    com: int,
    min_periods: int,
    min_corr_denom: float = 1e-14,
) -> tuple[np.ndarray, _EwmCorrState]:
    """Compute per-row EWM correlation matrices and return the final IIR state.

    Identical to `ewm_corr` but also returns the final filter
    memory as an `_EwmCorrState`.  Callers that need both the
    correlation tensor *and* the IIR state (e.g.
    `warmup_state`) should call this function once rather
    than calling `ewm_corr` and rerunning ``lfilter`` separately
    to extract the state.

    Args:
        data: Float array of shape ``(T, N)`` — typically volatility-adjusted
            log returns.
        com: EWM centre-of-mass.
        min_periods: Minimum joint-finite observations before a value is
            reported.
        min_corr_denom: Guard threshold for the correlation denominator.

    Returns:
        tuple: ``(result, state)`` where ``result`` has shape ``(T, N, N)``
        (see `ewm_corr`) and ``state`` is the final
        `_EwmCorrState`.
    """
    _t_len, n_assets = data.shape
    beta = com / (1.0 + com)

    fin = np.isfinite(data)
    xt_f = np.where(fin, data, 0.0)
    joint_fin = fin[:, :, np.newaxis] & fin[:, np.newaxis, :]

    v_x = xt_f[:, :, np.newaxis] * joint_fin
    v_x2 = (xt_f * xt_f)[:, :, np.newaxis] * joint_fin
    v_xy = xt_f[:, :, np.newaxis] * xt_f[:, np.newaxis, :]
    v_w = joint_fin.astype(np.float64)

    filt_a = np.array([1.0, -beta])
    s_x = lfilter([1.0], filt_a, v_x, axis=0)
    s_x2 = lfilter([1.0], filt_a, v_x2, axis=0)
    s_xy = lfilter([1.0], filt_a, v_xy, axis=0)
    s_w = lfilter([1.0], filt_a, v_w, axis=0)

    count = np.cumsum(joint_fin, axis=0)

    # Final IIR state: for b=[1], a=[1,-beta] in direct-form-II-transposed,
    # the filter memory after sample y[-1] is zf[0] = beta * y[-1].
    # Shaped (1, N, N) to match the zi format expected by lfilter.
    iir_state = _EwmCorrState(
        corr_zi_x=(beta * s_x[-1])[np.newaxis],
        corr_zi_x2=(beta * s_x2[-1])[np.newaxis],
        corr_zi_xy=(beta * s_xy[-1])[np.newaxis],
        corr_zi_w=(beta * s_w[-1])[np.newaxis],
        count=count[-1].astype(np.int64),
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        pos_w = s_w > 0
        ewm_x = np.where(pos_w, s_x / s_w, np.nan)
        ewm_y = np.where(pos_w, s_x.swapaxes(1, 2) / s_w, np.nan)
        ewm_x2 = np.where(pos_w, s_x2 / s_w, np.nan)
        ewm_y2 = np.where(pos_w, s_x2.swapaxes(1, 2) / s_w, np.nan)
        ewm_xy = np.where(pos_w, s_xy / s_w, np.nan)

    var_x = np.maximum(ewm_x2 - ewm_x * ewm_x, 0.0)
    var_y = np.maximum(ewm_y2 - ewm_y * ewm_y, 0.0)
    denom = np.sqrt(var_x * var_y)
    cov = ewm_xy - ewm_x * ewm_y

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom > min_corr_denom, cov / denom, np.nan)

    result = np.clip(result, -1.0, 1.0)
    result[count < min_periods] = np.nan

    diag_idx = np.arange(n_assets)
    diag_count = count[:, diag_idx, diag_idx]
    result[:, diag_idx, diag_idx] = np.where(diag_count >= min_periods, 1.0, np.nan)

    # Enforce symmetry without allocating an extra (T, N, N) temporary:
    # average only over strict triangle pairs and mirror in place.
    tril_i, tril_j = np.tril_indices(n_assets, k=-1)
    upper_vals = result[:, tril_j, tril_i]
    lower_vals = result[:, tril_i, tril_j]
    avg_vals = 0.5 * (upper_vals + lower_vals)
    result[:, tril_i, tril_j] = avg_vals
    result[:, tril_j, tril_i] = avg_vals

    return result, iir_state


def ewm_corr(
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
    result, _ = _ewm_corr_with_final_state(data, com, min_periods, min_corr_denom)
    return result
