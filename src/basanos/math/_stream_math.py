"""Pure numerical helpers for the incremental (streaming) optimiser.

These functions carry no stream state — they are factored out of
:mod:`basanos.math._stream` so the EWMA recurrences and the per-step input
validation can be read, reused, and tested in isolation.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


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


def _ewm_vol_accumulators_from_batch(
    returns: np.ndarray,
    beta: float,
    beta_sq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute final EWMA volatility accumulators from a batch of returns.

    Implements the same IIR recurrence as ``BasanosStream.step`` but
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
    ``BasanosStream.from_warmup`` for both the log-return (``vola_*``)
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
