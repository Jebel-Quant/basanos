"""Per-step position solvers for the incremental (streaming) optimiser.

Two entry points — `solve_sliding_window_position` and `solve_ewma_position` —
each take the resolved step inputs plus the current `_StreamState` and return a
``(cash_position, status)`` pair.  They hold no reference to the
`BasanosStream` façade, so the two covariance modes can be read and tested
independently of the streaming loop that drives them.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import polars as pl
from cvx.linalg import SingularMatrixError, cov_to_corr
from cvx.linalg.covariance.ewm_cov import ewm_covariance

from ._config import BasanosConfig, SlidingWindowConfig
from ._engine_solve import MatrixBundle, SolveStatus, _SolveMixin
from ._factor_model import FactorModel
from ._signal import shrink2id
from ._stream_state import _StreamState

_logger = logging.getLogger(__name__)


def solve_sliding_window_position(
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
    new_cash_pos = np.full(n_assets, np.nan, dtype=float)
    status = SolveStatus.DEGENERATE
    sw_config = cast(SlidingWindowConfig, cfg.covariance_config)
    if not mask.any():
        return new_cash_pos, status

    win_w = sw_config.window
    win_k = sw_config.n_factors
    sw_ret_buf = cast(np.ndarray, state.sw_ret_buf)
    window_ret = np.where(
        np.isfinite(sw_ret_buf[:, mask]),
        sw_ret_buf[:, mask],
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


def solve_ewma_position(
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
        # keys are the integer ``t`` index values built from ``range(t)`` above
        latest = max(cov_dict, key=lambda k: cast("int", k))
        corr = cov_to_corr(cov_dict[latest], cfg.min_corr_denom)
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
