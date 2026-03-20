"""Correlation-aware risk position optimizer (Basanos).

This module provides utilities to compute correlation-adjusted risk positions
from price data and expected-return signals. It relies on volatility-adjusted
returns to estimate a dynamic correlation matrix (via EWM), applies shrinkage
towards identity, and solves a normalized linear system per timestamp to
obtain stable positions.

Performance characteristics
---------------------------
Let *N* be the number of assets and *T* the number of timestamps.

**Computational complexity**

+----------------------------------+------------------+--------------------------------------+
| Operation                        | Complexity       | Bottleneck                           |
+==================================+==================+======================================+
| EWM volatility (``ret_adj``,     | O(T·N)           | Linear in both T and N; negligible   |
| ``vola``)                        |                  |                                      |
+----------------------------------+------------------+--------------------------------------+
| EWM correlation (``cor``)        | O(T·N²)          | ``lfilter`` over all N² asset pairs  |
|                                  |                  | simultaneously                       |
+----------------------------------+------------------+--------------------------------------+
| Linear solve per timestamp       | O(N³)            | Cholesky / LU per row in             |
| (``cash_position``)              | * T solves       | ``cash_position``                    |
+----------------------------------+------------------+--------------------------------------+

**Memory usage** (peak, approximate)

``_ewm_corr_numpy`` allocates roughly **14 float64 arrays** of shape
``(T, N, N)`` at peak (input sequences, IIR filter outputs, EWM components,
and the result tensor).  Peak RAM ≈ **112 * T * N²** bytes.  Typical
working sizes on a 16 GB machine:

+--------+--------------------------+------------------------------------+
| N      | T (daily rows)           | Peak memory (approx.)              |
+========+==========================+====================================+
| 50     | 252 (~1 yr)              | ~70 MB                             |
+--------+--------------------------+------------------------------------+
| 100    | 252 (~1 yr)              | ~280 MB                            |
+--------+--------------------------+------------------------------------+
| 100    | 2 520 (~10 yr)           | ~2.8 GB                            |
+--------+--------------------------+------------------------------------+
| 200    | 2 520 (~10 yr)           | ~11 GB                             |
+--------+--------------------------+------------------------------------+
| 500    | 2 520 (~10 yr)           | ~70 GB ⚠ exceeds typical RAM       |
+--------+--------------------------+------------------------------------+

**Practical limits (daily data)**

* **≤ 150 assets, ≤ 5 years** — well within reach on an 8 GB laptop.
* **≤ 250 assets, ≤ 10 years** — requires ~11-12 GB; feasible on a 16 GB
  workstation.
* **> 500 assets with multi-year history** — peak memory exceeds 16 GB;
  reduce the time range or switch to a chunked / streaming approach.
* **> 1 000 assets** — the O(N³) per-solve cost alone makes real-time
  optimization impractical even with adequate RAM.

See ``BENCHMARKS.md`` for measured wall-clock timings across representative
dataset sizes.

Internal structure
------------------
The implementation is split across focused private modules to reduce
God-Object complexity:

* :mod:`basanos.math._config` — :class:`BasanosConfig` and all
  covariance-mode configuration classes.
* :mod:`basanos.math._engine_diagnostics` — :class:`_DiagnosticsMixin`
  (condition number, effective rank, solver residual, signal utilisation).
* :mod:`basanos.math._engine_ic` — :class:`_SignalEvaluatorMixin`
  (IC, Rank IC, ICIR, and summary statistics).
* This module — ``_ewm_corr_numpy``, core solve/position logic, and the
  :class:`BasanosEngine` facade that wires everything together.
"""

import dataclasses
import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from scipy.signal import lfilter

from ..analytics import Portfolio
from ..exceptions import (
    ColumnMismatchError,
    ExcessiveNullsError,
    MissingDateColumnError,
    MonotonicPricesError,
    NonPositivePricesError,
    ShapeMismatchError,
    SingularMatrixError,
)
from ._config import (
    BasanosConfig,
    CovarianceConfig,
    CovarianceMode,
    EwmaShrinkConfig,
    SlidingWindowConfig,
)
from ._engine_diagnostics import _DiagnosticsMixin
from ._engine_ic import _SignalEvaluatorMixin
from ._factor_model import FactorModel
from ._linalg import inv_a_norm, solve
from ._signal import shrink2id, vol_adj

if TYPE_CHECKING:
    from ._config_report import ConfigReport

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-export config symbols so ``from basanos.math.optimizer import …`` keeps
# working for existing callers.
# ---------------------------------------------------------------------------
__all__ = [
    "BasanosConfig",
    "BasanosEngine",
    "CovarianceConfig",
    "CovarianceMode",
    "EwmaShrinkConfig",
    "SlidingWindowConfig",
]


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


@dataclasses.dataclass(frozen=True)
class BasanosEngine(_DiagnosticsMixin, _SignalEvaluatorMixin):
    """Engine to compute correlation matrices and optimize risk positions.

    Encapsulates price data and configuration to build EWM-based
    correlations, apply shrinkage, and solve for normalized positions.

    The engine is organized into focused concerns, each handled by a
    dedicated sub-module:

    * **Core data access** — :attr:`assets`, :attr:`ret_adj`, :attr:`vola`,
      :attr:`cor`, :attr:`cor_tensor`
    * **Solve / position logic** — :attr:`cash_position`,
      :attr:`position_status`, :attr:`risk_position`,
      :attr:`position_leverage` (this module)
    * **Portfolio and performance** — :attr:`portfolio`,
      :attr:`naive_sharpe`, :meth:`sharpe_at_shrink`,
      :meth:`sharpe_at_window_factors` (this module)
    * **Matrix diagnostics** — :attr:`condition_number`,
      :attr:`effective_rank`, :attr:`solver_residual`,
      :attr:`signal_utilisation`
      (:mod:`basanos.math._engine_diagnostics`)
    * **Signal evaluation** — :attr:`ic`, :attr:`rank_ic`, :attr:`ic_mean`,
      :attr:`ic_std`, :attr:`icir`, :attr:`rank_ic_mean`,
      :attr:`rank_ic_std`
      (:mod:`basanos.math._engine_ic`)

    Attributes:
        prices: Polars DataFrame of price levels per asset over time.  Must
            contain a ``'date'`` column and at least one numeric asset column
            with strictly positive values that are not monotonically
            non-decreasing or non-increasing (i.e. they must vary in sign).
        mu: Polars DataFrame of expected-return signals aligned with *prices*.
            Must share the same shape and column names as *prices*.
        cfg: Immutable :class:`BasanosConfig` controlling EWMA half-lives,
            clipping, shrinkage intensity, and AUM.

    Examples:
        Build an engine with two synthetic assets over 30 days and inspect the
        optimized positions and diagnostic properties.

        >>> import numpy as np
        >>> import polars as pl
        >>> from basanos.math import BasanosConfig, BasanosEngine
        >>> dates = list(range(30))
        >>> rng = np.random.default_rng(42)
        >>> prices = pl.DataFrame({
        ...     "date": dates,
        ...     "A": np.cumprod(1 + rng.normal(0.001, 0.02, 30)) * 100.0,
        ...     "B": np.cumprod(1 + rng.normal(0.001, 0.02, 30)) * 150.0,
        ... })
        >>> mu = pl.DataFrame({
        ...     "date": dates,
        ...     "A": rng.normal(0.0, 0.5, 30),
        ...     "B": rng.normal(0.0, 0.5, 30),
        ... })
        >>> cfg = BasanosConfig(vola=5, corr=10, clip=2.0, shrink=0.5, aum=1_000_000)
        >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
        >>> engine.assets
        ['A', 'B']
        >>> engine.cash_position.shape
        (30, 3)
        >>> engine.position_leverage.columns
        ['date', 'leverage']
    """

    prices: pl.DataFrame
    mu: pl.DataFrame
    cfg: BasanosConfig

    def __post_init__(self) -> None:
        """Validate basic invariants right after initialization.

        Ensures both ``prices`` and ``mu`` contain a ``'date'`` column and
        share identical shapes/columns, which downstream computations rely on.
        Also checks for data quality issues that would cause silent failures
        downstream: non-positive prices, excessive NaN values, and monotonic
        (non-varying) price series.
        """
        # ensure 'date' column exists in prices before any other validation
        if "date" not in self.prices.columns:
            raise MissingDateColumnError("prices")

        # ensure 'date' column exists in mu as well (kept for symmetry and downstream assumptions)
        if "date" not in self.mu.columns:
            raise MissingDateColumnError("mu")

        # check that prices and mu have the same shape
        if self.prices.shape != self.mu.shape:
            raise ShapeMismatchError(self.prices.shape, self.mu.shape)

        # check that the columns of prices and mu are identical
        if not set(self.prices.columns) == set(self.mu.columns):
            raise ColumnMismatchError(self.prices.columns, self.mu.columns)

        # check for non-positive prices: log returns require strictly positive prices
        for asset in self.assets:
            col = self.prices[asset].drop_nulls()
            if col.len() > 0 and (col <= 0).any():
                raise NonPositivePricesError(asset)

        # check for excessive NaN values: more than cfg.max_nan_fraction null in any asset column
        n_rows = self.prices.height
        if n_rows > 0:
            for asset in self.assets:
                nan_frac = self.prices[asset].null_count() / n_rows
                if nan_frac > self.cfg.max_nan_fraction:
                    raise ExcessiveNullsError(asset, nan_frac, self.cfg.max_nan_fraction)

        # check for monotonic price series: a strictly non-decreasing or non-increasing
        # series has no variance in its return sign, indicating malformed or synthetic data
        for asset in self.assets:
            col = self.prices[asset].drop_nulls()
            if col.len() > 2:
                diffs = col.diff().drop_nulls()
                if (diffs >= 0).all() or (diffs <= 0).all():
                    raise MonotonicPricesError(asset)

        # warn when the dataset is too short to benefit from the sliding window
        if self.cfg.covariance_mode == CovarianceMode.sliding_window and self.cfg.window is not None:
            w: int = self.cfg.window
            n_rows = self.prices.height
            if n_rows < 2 * w:
                _logger.warning(
                    "Dataset length (%d rows) is less than 2 * window (%d). "
                    "The first %d timestamps will yield zero positions during warm-up; "
                    "consider using a longer history or reducing 'window'.",
                    n_rows,
                    2 * w,
                    w - 1,
                )

    # ------------------------------------------------------------------
    # Core data-access properties
    # ------------------------------------------------------------------

    @property
    def assets(self) -> list[str]:
        """List asset column names (numeric columns excluding 'date')."""
        return [c for c in self.prices.columns if c != "date" and self.prices[c].dtype.is_numeric()]

    @property
    def ret_adj(self) -> pl.DataFrame:
        """Return per-asset volatility-adjusted log returns clipped by cfg.clip.

        Uses an EWMA volatility estimate with lookback ``cfg.vola`` to
        standardize log returns for each numeric asset column.
        """
        return self.prices.with_columns(
            [vol_adj(pl.col(asset), vola=self.cfg.vola, clip=self.cfg.clip) for asset in self.assets]
        )

    @property
    def vola(self) -> pl.DataFrame:
        """Per-asset EWMA volatility of percentage returns.

        Computes percent changes for each numeric asset column and applies an
        exponentially weighted standard deviation using the lookback specified
        by ``cfg.vola``. The result is a DataFrame aligned with ``self.prices``
        whose numeric columns hold per-asset volatility estimates.
        """
        return self.prices.with_columns(
            pl.col(asset)
            .pct_change()
            .ewm_std(com=self.cfg.vola - 1, adjust=True, min_samples=self.cfg.vola)
            .alias(asset)
            for asset in self.assets
        )

    @property
    def cor(self) -> dict[object, np.ndarray]:
        """Compute per-timestamp EWM correlation matrices.

        Builds volatility-adjusted returns for all assets, computes an
        exponentially weighted correlation using a pure NumPy implementation
        (with window ``cfg.corr``), and returns a mapping from each timestamp
        to the corresponding correlation matrix as a NumPy array.

        Returns:
            dict: Mapping ``date -> np.ndarray`` of shape (n_assets, n_assets).

        Performance:
            Delegates to :func:`_ewm_corr_numpy`, which is O(T·N²) in both
            time and memory.  The returned dict holds *T* references into the
            result tensor (one N*N view per date); no extra copies are made.
            For large *N* or *T*, prefer ``cor_tensor`` to keep a single
            contiguous array rather than building a Python dict.
        """
        index = self.prices["date"]
        ret_adj_np = self.ret_adj.select(self.assets).to_numpy()
        tensor = _ewm_corr_numpy(
            ret_adj_np,
            com=self.cfg.corr,
            min_periods=self.cfg.corr,
            min_corr_denom=self.cfg.min_corr_denom,
        )
        return {index[t]: tensor[t] for t in range(len(index))}

    @property
    def cor_tensor(self) -> np.ndarray:
        """Return all correlation matrices stacked as a 3-D tensor.

        Converts the per-timestamp correlation dict (see :py:attr:`cor`) into a
        single contiguous NumPy array so that the full history can be saved to
        a flat ``.npy`` file with :func:`numpy.save` and reloaded with
        :func:`numpy.load`.

        Returns:
            np.ndarray: Array of shape ``(T, N, N)`` where *T* is the number of
            timestamps and *N* the number of assets.  ``tensor[t]`` is the
            correlation matrix for the *t*-th date (same ordering as
            ``self.prices["date"]``).

        Examples:
            >>> import tempfile, pathlib
            >>> import numpy as np
            >>> import polars as pl
            >>> from basanos.math.optimizer import BasanosConfig, BasanosEngine
            >>> dates = pl.Series("date", list(range(100)))
            >>> rng0 = np.random.default_rng(0).lognormal(size=100)
            >>> rng1 = np.random.default_rng(1).lognormal(size=100)
            >>> prices = pl.DataFrame({"date": dates, "A": rng0, "B": rng1})
            >>> rng2 = np.random.default_rng(2).normal(size=100)
            >>> rng3 = np.random.default_rng(3).normal(size=100)
            >>> mu = pl.DataFrame({"date": dates, "A": rng2, "B": rng3})
            >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
            >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
            >>> tensor = engine.cor_tensor
            >>> with tempfile.TemporaryDirectory() as td:
            ...     path = pathlib.Path(td) / "cor.npy"
            ...     np.save(path, tensor)
            ...     loaded = np.load(path)
            >>> np.testing.assert_array_equal(tensor, loaded)
        """
        return np.stack(list(self.cor.values()), axis=0)

    # ------------------------------------------------------------------
    # Internal solve helpers
    # ------------------------------------------------------------------

    def _iter_matrices(self):
        r"""Yield ``(i, t, mask, matrix)`` for every timestamp.

        ``matrix`` is the effective :math:`(n_{\text{sub}},\ n_{\text{sub}})`
        correlation matrix for the active assets (those with finite prices at
        timestamp *t*).  Yields ``None`` when no valid matrix is available
        (e.g., before the warm-up period has elapsed or when no assets have
        finite prices).

        The behaviour depends on :attr:`BasanosConfig.covariance_config`:

        * :class:`EwmaShrinkConfig`:  Applies :func:`~basanos.math._signal.shrink2id` to
          the EWMA correlation matrix (same computation as
          :attr:`cash_position`).
        * :class:`SlidingWindowConfig`: Builds a
          :class:`~basanos.math._factor_model.FactorModel` from the last
          ``cfg.covariance_config.window`` rows of vol-adjusted returns and returns its
          :attr:`~basanos.math._factor_model.FactorModel.covariance`.

        Yields:
            tuple: ``(i, t, mask, matrix)`` where

            * ``i`` (*int*): Row index into ``self.prices``.
            * ``t``: Timestamp value from ``self.prices["date"]``.
            * ``mask`` (*np.ndarray[bool]*): Shape ``(n_assets,)``; ``True``
              for assets with finite prices at row *i*.
            * ``matrix`` (*np.ndarray | None*): Shape
              ``(mask.sum(), mask.sum())`` or ``None``.
        """
        assets = self.assets
        prices_num = self.prices.select(assets).to_numpy()
        dates = self.prices["date"].to_list()

        if isinstance(self.cfg.covariance_config, EwmaShrinkConfig):
            cor = self.cor
            for i, t in enumerate(dates):
                mask = np.isfinite(prices_num[i])
                if not mask.any():
                    yield i, t, mask, None
                    continue
                corr_n = cor[t]
                matrix = shrink2id(corr_n, lamb=self.cfg.shrink)[np.ix_(mask, mask)]
                yield i, t, mask, matrix
        else:
            sw_config = cast(SlidingWindowConfig, self.cfg.covariance_config)
            win_w: int = sw_config.window
            win_k: int = sw_config.n_factors
            ret_adj_np = self.ret_adj.select(assets).to_numpy()
            for i, t in enumerate(dates):
                mask = np.isfinite(prices_num[i])
                if not mask.any() or i + 1 < win_w:
                    yield i, t, mask, None
                    continue
                window_ret = ret_adj_np[i + 1 - win_w : i + 1][:, mask]
                window_ret = np.where(np.isfinite(window_ret), window_ret, 0.0)
                n_sub = int(mask.sum())
                k_eff = min(win_k, win_w, n_sub)
                try:
                    fm = FactorModel.from_returns(window_ret, k=k_eff)
                    yield i, t, mask, fm.covariance
                except (np.linalg.LinAlgError, ValueError) as exc:
                    _logger.warning("Factor model fit failed at t=%s: %s", t, exc)
                    yield i, t, mask, None

    def _iter_solve(self):
        r"""Yield ``(i, t, mask, pos_or_none, status)`` for every timestamp.

        This is the single authoritative implementation of the per-timestamp
        position-determination logic shared by :attr:`cash_position` and
        :attr:`position_status`.  Extracting it here eliminates the DRY
        violation where both properties previously duplicated the mask
        computation, covariance dispatch, :math:`\mu` handling, denominator
        check, and linear-solve logic.

        Yields:
            tuple: ``(i, t, mask, pos_or_none, status)`` where

            * ``i`` (*int*): Row index into ``self.prices``.
            * ``t``: Timestamp value from ``self.prices["date"]``.
            * ``mask`` (*np.ndarray[bool]*): Shape ``(n_assets,)``; ``True``
              for assets with finite prices at row *i*.
            * ``pos_or_none`` (*np.ndarray | None*): Per-active-asset position
              vector **before** ``profit_variance`` scaling.  The value and its
              downstream effect depend on ``status`` as follows:

              .. list-table::
                 :header-rows: 1

                 * - ``status``
                   - ``pos_or_none``
                   - Downstream effect in :attr:`cash_position`
                 * - ``'warmup'``
                   - ``None``
                   - Positions stay ``NaN`` — insufficient history.
                 * - ``'zero_signal'``
                   - ``np.zeros(n_active)``
                   - Positions written as ``0`` for all active assets.
                 * - ``'degenerate'``
                   - ``np.zeros(n_active)`` (empty when no prices)
                   - Positions written as ``0`` for active assets; rows with
                     no finite prices have an empty mask so all positions
                     remain ``NaN`` as a natural consequence.
                 * - ``'valid'``
                   - ``np.ndarray`` of shape ``(n_active,)``
                   - Solved positions written for all active assets.

              ``None`` is yielded **only** for ``'warmup'`` rows.  All other
              statuses yield an ``np.ndarray`` (possibly zero-length when
              ``mask`` is all-``False``), so consumers can branch solely on
              ``pos_or_none is None`` to detect the warmup case without
              inspecting ``status``.

            * ``status`` (*str*): One of ``'warmup'``, ``'zero_signal'``,
              ``'degenerate'``, or ``'valid'``.
        """
        assets = self.assets
        prices_num = self.prices.select(assets).to_numpy()
        mu_np = self.mu.select(assets).to_numpy()
        dates = self.prices["date"].to_list()

        if isinstance(self.cfg.covariance_config, EwmaShrinkConfig):
            cor = self.cor
            for i, t in enumerate(dates):
                mask = np.isfinite(prices_num[i])
                if not mask.any():
                    yield i, t, mask, np.zeros(0), "degenerate"
                    continue
                corr_n = cor[t]
                matrix = shrink2id(corr_n, lamb=self.cfg.shrink)[np.ix_(mask, mask)]
                expected_mu = np.nan_to_num(mu_np[i][mask])
                if np.allclose(expected_mu, 0.0):
                    yield i, t, mask, np.zeros_like(expected_mu), "zero_signal"
                    continue
                try:
                    denom = inv_a_norm(expected_mu, matrix)
                except SingularMatrixError:
                    denom = float("nan")
                if not np.isfinite(denom) or denom <= self.cfg.denom_tol:
                    _logger.warning(
                        "Positions zeroed at t=%s: normalisation denominator is degenerate "
                        "(denom=%s, denom_tol=%s). Check signal magnitude and covariance matrix.",
                        t,
                        denom,
                        self.cfg.denom_tol,
                        extra={
                            "context": {
                                "t": str(t),
                                "denom": denom,
                                "denom_tol": self.cfg.denom_tol,
                            }
                        },
                    )
                    yield i, t, mask, np.zeros_like(expected_mu), "degenerate"
                    continue
                try:
                    pos = solve(matrix, expected_mu) / denom
                except SingularMatrixError:
                    yield i, t, mask, np.zeros_like(expected_mu), "degenerate"
                    continue
                yield i, t, mask, pos, "valid"
        else:
            sw_config = cast(SlidingWindowConfig, self.cfg.covariance_config)
            win_w: int = sw_config.window
            win_k: int = sw_config.n_factors
            ret_adj_np = self.ret_adj.select(assets).to_numpy()
            for i, t in enumerate(dates):
                mask = np.isfinite(prices_num[i])
                if not mask.any():
                    yield i, t, mask, np.zeros(0), "degenerate"
                    continue
                if i + 1 < win_w:
                    yield i, t, mask, None, "warmup"
                    continue
                window_ret = ret_adj_np[i + 1 - win_w : i + 1][:, mask]
                window_ret = np.where(np.isfinite(window_ret), window_ret, 0.0)
                n_sub = int(mask.sum())
                k_eff = min(win_k, win_w, n_sub)
                try:
                    fm = FactorModel.from_returns(window_ret, k=k_eff)
                except (np.linalg.LinAlgError, ValueError) as exc:
                    _logger.debug("Sliding window SVD failed at t=%s: %s", t, exc)
                    yield i, t, mask, np.zeros(n_sub), "degenerate"
                    continue
                expected_mu = np.nan_to_num(mu_np[i][mask])
                if np.allclose(expected_mu, 0.0):
                    yield i, t, mask, np.zeros(n_sub), "zero_signal"
                    continue
                try:
                    x = fm.solve(expected_mu)
                    denom = float(np.sqrt(max(0.0, float(np.dot(expected_mu, x)))))
                except (np.linalg.LinAlgError, ValueError) as exc:
                    _logger.warning("Woodbury solve failed at t=%s: %s", t, exc)
                    yield i, t, mask, np.zeros(n_sub), "degenerate"
                    continue
                if not np.isfinite(denom) or denom <= self.cfg.denom_tol:
                    _logger.warning(
                        "Positions zeroed at t=%s (sliding_window): normalisation "
                        "denominator is degenerate (denom=%s, denom_tol=%s).",
                        t,
                        denom,
                        self.cfg.denom_tol,
                    )
                    yield i, t, mask, np.zeros(n_sub), "degenerate"
                    continue
                yield i, t, mask, x / denom, "valid"

    # ------------------------------------------------------------------
    # Position properties
    # ------------------------------------------------------------------

    @property
    def cash_position(self) -> pl.DataFrame:
        r"""Optimize correlation-aware risk positions for each timestamp.

        Supports two covariance modes controlled by ``cfg.covariance_config``:

        * :class:`EwmaShrinkConfig` (default): Computes EWMA correlations, applies
          linear shrinkage toward the identity, and solves a normalised linear
          system :math:`C\,x = \mu` per timestamp via Cholesky / LU.

        * :class:`SlidingWindowConfig`: At each timestamp uses the
          ``cfg.covariance_config.window`` most recent vol-adjusted returns to fit a
          rank-``cfg.covariance_config.n_factors`` factor model via truncated SVD and
          solves the system via the Woodbury identity at :math:`O(k^3 + kn)` rather
          than :math:`O(n^3)` per step.

        Non-finite or ill-posed cases yield zero positions for safety.

        Returns:
            pl.DataFrame: DataFrame with columns ['date'] + asset names containing
            the per-timestamp cash positions (risk divided by EWMA volatility).

        Performance:
            For ``ewma_shrink``: dominant cost is ``self.cor`` (O(T·N²) time,
            O(T·N²) memory — see :func:`_ewm_corr_numpy`).  The per-timestamp
            linear solve adds O(N³) per row.

            For ``sliding_window``: O(T·W·N·k) for sliding SVDs plus
            O(T·(k³ + kN)) for Woodbury solves.  Memory is O(W·N) per step,
            independent of T.
        """
        assets = self.assets

        # Compute risk positions row-by-row using _iter_solve for the solve logic.
        prices_num = self.prices.select(assets).to_numpy()
        returns_num = np.zeros_like(prices_num, dtype=float)
        returns_num[1:] = prices_num[1:] / prices_num[:-1] - 1.0

        risk_pos_np = np.full_like(prices_num, fill_value=np.nan, dtype=float)
        cash_pos_np = np.full_like(prices_num, fill_value=np.nan, dtype=float)
        vola_np = self.vola.select(assets).to_numpy()

        profit_variance = self.cfg.profit_variance_init
        lamb = self.cfg.profit_variance_decay

        for i, _t, mask, pos, _status in self._iter_solve():
            # Compute profit contribution using only finite returns and available positions.
            # This must happen for every row (including warmup / degenerate) so that the
            # profit_variance EMA is updated from the correct previous-step cash positions.
            if i > 0:
                ret_mask = np.isfinite(returns_num[i]) & mask
                # Profit at time i comes from yesterday's cash position applied to today's returns
                if ret_mask.any():
                    with np.errstate(invalid="ignore"):
                        cash_pos_np[i - 1] = risk_pos_np[i - 1] / vola_np[i - 1]
                    lhs = np.nan_to_num(cash_pos_np[i - 1, ret_mask], nan=0.0)
                    rhs = np.nan_to_num(returns_num[i, ret_mask], nan=0.0)
                    profit = lhs @ rhs
                    profit_variance = lamb * profit_variance + (1 - lamb) * profit**2

            if pos is not None:
                risk_pos_np[i, mask] = pos / profit_variance
                with np.errstate(invalid="ignore"):
                    cash_pos_np[i, mask] = risk_pos_np[i, mask] / vola_np[i, mask]

        # Build Polars DataFrame for cash positions (numeric columns only)
        cash_position = self.prices.with_columns(
            [(pl.lit(cash_pos_np[:, i]).alias(asset)) for i, asset in enumerate(assets)]
        )

        return cash_position

    @property
    def position_status(self) -> pl.DataFrame:
        """Per-timestamp reason code explaining each :attr:`cash_position` row.

        Labels every row with exactly one of four string codes:

        * ``'warmup'``: Insufficient history for the sliding-window
          covariance mode (``i + 1 < cfg.covariance_config.window``).
          Positions are ``NaN`` for all assets at this timestamp.
        * ``'zero_signal'``: The expected-return vector ``mu`` was
          all-zeros (or all-NaN) at this timestamp; the optimizer
          short-circuited and returned zero positions without solving.
        * ``'degenerate'``: The normalisation denominator was non-finite
          or below ``cfg.denom_tol``, the Cholesky / Woodbury solve
          failed, or no asset had a finite price; positions were zeroed
          for safety.
        * ``'valid'``: The linear system was solved successfully and
          positions are non-trivially non-zero.

        The codes map one-to-one onto the three NaN / zero cases
        described in the issue and allow downstream consumers (backtests,
        risk monitors) to distinguish data gaps from signal silence from
        numerical ill-conditioning without re-inspecting ``mu`` or the
        engine configuration.

        Returns:
            pl.DataFrame: Two-column DataFrame ``{'date': ..., 'status': ...}``
            with one row per timestamp.  The ``status`` column has
            ``Polars`` dtype ``String``.
        """
        statuses = [status for _i, _t, _mask, _pos, status in self._iter_solve()]
        return pl.DataFrame({"date": self.prices["date"], "status": pl.Series(statuses, dtype=pl.String)})

    @property
    def risk_position(self) -> pl.DataFrame:
        """Risk positions (before EWMA-volatility scaling) at each timestamp.

        Derives the un-volatility-scaled position by multiplying the cash
        position by the per-asset EWMA volatility.  Equivalently, this is
        the quantity solved by the correlation-adjusted linear system before
        dividing by ``vola``.

        Relationship to other properties::

            cash_position = risk_position / vola
            risk_position = cash_position * vola

        Returns:
            pl.DataFrame: DataFrame with columns ``['date'] + assets`` where
            each value is ``cash_position_i * vola_i`` at the given timestamp.
        """
        assets = self.assets
        cp_np = self.cash_position.select(assets).to_numpy()
        vola_np = self.vola.select(assets).to_numpy()
        with np.errstate(invalid="ignore"):
            risk_pos = cp_np * vola_np
        return self.prices.with_columns([pl.lit(risk_pos[:, i]).alias(asset) for i, asset in enumerate(assets)])

    @property
    def position_leverage(self) -> pl.DataFrame:
        """L1 norm of cash positions (gross leverage) at each timestamp.

        Sums the absolute values of all asset cash positions at each row.
        NaN positions are treated as zero (they contribute nothing to gross
        leverage).

        Returns:
            pl.DataFrame: Two-column DataFrame ``{'date': ..., 'leverage': ...}``
            where ``leverage`` is the L1 norm of the cash-position vector.
        """
        assets = self.assets
        cp_np = self.cash_position.select(assets).to_numpy()
        leverage = np.nansum(np.abs(cp_np), axis=1)
        return pl.DataFrame({"date": self.prices["date"], "leverage": pl.Series(leverage, dtype=pl.Float64)})

    # ------------------------------------------------------------------
    # Portfolio and performance
    # ------------------------------------------------------------------

    @property
    def portfolio(self) -> Portfolio:
        """Construct a Portfolio from the optimized cash positions.

        Converts the computed cash positions into a Portfolio using the
        configured AUM.

        Returns:
            Portfolio: Instance built from cash positions with AUM scaling.
        """
        cp = self.cash_position
        assets = [c for c in cp.columns if c != "date" and cp[c].dtype.is_numeric()]
        scaled = cp.with_columns(pl.col(a) * self.cfg.position_scale for a in assets)
        return Portfolio.from_cash_position(self.prices, scaled, aum=self.cfg.aum)

    def sharpe_at_shrink(self, shrink: float) -> float:
        r"""Return the annualised portfolio Sharpe ratio for the given shrinkage weight.

        Constructs a new :class:`BasanosEngine` with all parameters identical to
        ``self`` except that ``cfg.shrink`` is replaced by ``shrink``, then
        returns the annualised Sharpe ratio of the resulting portfolio.

        This is the canonical single-argument callable required by the benchmarks
        specification: ``f(λ) → Sharpe``.  Use it to sweep λ across ``[0, 1]``
        and measure whether correlation adjustment adds value over the
        signal-proportional baseline (λ = 0) or the unregularised limit (λ = 1).

        Corner cases:
            * **λ = 0** — the shrunk matrix equals the identity, so the
              optimiser treats all assets as uncorrelated and positions are
              purely signal-proportional (no correlation adjustment).
            * **λ = 1** — the raw EWMA correlation matrix is used without
              shrinkage.

        Args:
            shrink: Retention weight λ ∈ [0, 1].  See
                :attr:`BasanosConfig.shrink` for full documentation.

        Returns:
            Annualised Sharpe ratio of the portfolio returns as a ``float``.
            Returns ``float("nan")`` when the Sharpe ratio cannot be computed
            (e.g. zero-variance returns).

        Raises:
            ValidationError: When ``shrink`` is outside [0, 1] (delegated to
                :class:`BasanosConfig` field validation).

        Examples:
            >>> import numpy as np
            >>> import polars as pl
            >>> from basanos.math.optimizer import BasanosConfig, BasanosEngine
            >>> dates = pl.Series("date", list(range(200)))
            >>> rng = np.random.default_rng(0)
            >>> prices = pl.DataFrame({"date": dates, "A": rng.lognormal(size=200), "B": rng.lognormal(size=200)})
            >>> mu = pl.DataFrame({"date": dates, "A": rng.normal(size=200), "B": rng.normal(size=200)})
            >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
            >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
            >>> s = engine.sharpe_at_shrink(0.5)
            >>> isinstance(s, float)
            True
        """
        new_cfg = self.cfg.replace(shrink=shrink)
        engine = BasanosEngine(prices=self.prices, mu=self.mu, cfg=new_cfg)
        return float(engine.portfolio.stats.sharpe().get("returns") or float("nan"))

    def sharpe_at_window_factors(self, window: int, n_factors: int) -> float:
        r"""Return the annualised portfolio Sharpe ratio for the given sliding-window parameters.

        Constructs a new :class:`BasanosEngine` with ``covariance_mode`` set to
        ``"sliding_window"`` and the supplied ``window`` / ``n_factors``, keeping
        all other configuration identical to ``self``.

        Use this method to sweep ``(W, k)`` and compare the sliding-window
        estimator against the EWMA baseline (via :meth:`sharpe_at_shrink`).

        Args:
            window: Rolling window length :math:`W \geq 1`.
                Rule of thumb: :math:`W \geq 2 \cdot n_{\text{assets}}`.
            n_factors: Number of latent factors :math:`k \geq 1`.

        Returns:
            Annualised Sharpe ratio of the portfolio returns as a ``float``.
            Returns ``float("nan")`` when the Sharpe ratio cannot be computed
            (e.g. not enough history to fill the first window).

        Raises:
            ValidationError: When ``window`` or ``n_factors`` fail field
                constraints (delegated to :class:`BasanosConfig`).

        Examples:
            >>> import numpy as np
            >>> import polars as pl
            >>> from basanos.math.optimizer import BasanosConfig, BasanosEngine
            >>> dates = pl.Series("date", list(range(200)))
            >>> rng = np.random.default_rng(0)
            >>> prices = pl.DataFrame({"date": dates, "A": rng.lognormal(size=200), "B": rng.lognormal(size=200)})
            >>> mu = pl.DataFrame({"date": dates, "A": rng.normal(size=200), "B": rng.normal(size=200)})
            >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
            >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
            >>> s = engine.sharpe_at_window_factors(window=40, n_factors=2)
            >>> isinstance(s, float)
            True
        """
        new_cfg = self.cfg.replace(
            covariance_config=SlidingWindowConfig(window=window, n_factors=n_factors),
        )
        engine = BasanosEngine(prices=self.prices, mu=self.mu, cfg=new_cfg)
        return float(engine.portfolio.stats.sharpe().get("returns") or float("nan"))

    @property
    def naive_sharpe(self) -> float:
        r"""Sharpe ratio of the naïve equal-weight signal (μ = 1 for every asset/timestamp).

        Replaces the expected-return signal ``mu`` with a constant matrix of
        ones, then runs the optimiser with the current configuration and returns
        the annualised Sharpe ratio of the resulting portfolio.

        This provides the baseline answer to *"does the signal add value?"*:
        a real signal should produce a higher Sharpe than the naïve benchmark.
        Combined with :meth:`sharpe_at_shrink`, this yields a three-way
        comparison:

        +--------------------+----------------------------------------------+
        | Benchmark          | What it measures                             |
        +====================+==============================================+
        | ``naive_sharpe``   | No signal skill; pure correlation routing   |
        +--------------------+----------------------------------------------+
        | ``sharpe_at_shrink(0.0)`` | Signal skill, no correlation adj.  |
        +--------------------+----------------------------------------------+
        | ``sharpe_at_shrink(cfg.shrink)`` | Signal + correlation adj.  |
        +--------------------+----------------------------------------------+

        Returns:
            Annualised Sharpe ratio of the equal-weight portfolio as a ``float``.
            Returns ``float("nan")`` when the Sharpe ratio cannot be computed.

        Examples:
            >>> import numpy as np
            >>> import polars as pl
            >>> from basanos.math.optimizer import BasanosConfig, BasanosEngine
            >>> dates = pl.Series("date", list(range(200)))
            >>> rng = np.random.default_rng(0)
            >>> prices = pl.DataFrame({"date": dates, "A": rng.lognormal(size=200), "B": rng.lognormal(size=200)})
            >>> mu = pl.DataFrame({"date": dates, "A": rng.normal(size=200), "B": rng.normal(size=200)})
            >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
            >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
            >>> s = engine.naive_sharpe
            >>> isinstance(s, float)
            True
        """
        naive_mu = self.mu.with_columns(pl.lit(1.0).alias(asset) for asset in self.assets)
        engine = BasanosEngine(prices=self.prices, mu=naive_mu, cfg=self.cfg)
        return float(engine.portfolio.stats.sharpe().get("returns") or float("nan"))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def config_report(self) -> "ConfigReport":
        """Return a :class:`~basanos.math._config_report.ConfigReport` facade for this engine.

        Returns a :class:`~basanos.math._config_report.ConfigReport` that
        includes the full **lambda-sweep chart** — an interactive plot of the
        annualised Sharpe ratio as :attr:`~BasanosConfig.shrink` (λ) is swept
        across [0, 1] — in addition to the parameter table, shrinkage-guidance
        table, and theory section available from
        :attr:`BasanosConfig.report`.

        Returns:
            basanos.math._config_report.ConfigReport: Report facade with
            ``to_html()`` and ``save()`` methods.

        Examples:
            >>> import numpy as np
            >>> import polars as pl
            >>> from basanos.math.optimizer import BasanosConfig, BasanosEngine
            >>> dates = pl.Series("date", list(range(200)))
            >>> rng = np.random.default_rng(0)
            >>> prices = pl.DataFrame({"date": dates, "A": rng.lognormal(size=200), "B": rng.lognormal(size=200)})
            >>> mu = pl.DataFrame({"date": dates, "A": rng.normal(size=200), "B": rng.normal(size=200)})
            >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
            >>> engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
            >>> report = engine.config_report
            >>> html = report.to_html()
            >>> "Lambda" in html
            True
        """
        from ._config_report import ConfigReport

        return ConfigReport(config=self.cfg, engine=self)
