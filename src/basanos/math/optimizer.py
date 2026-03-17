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
"""

import dataclasses
import logging

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from scipy.signal import lfilter
from scipy.stats import spearmanr

from ..analytics import Portfolio
from ..exceptions import (
    ColumnMismatchError,
    ExcessiveNullsError,
    MissingDateColumnError,
    MonotonicPricesError,
    NonPositivePricesError,
    ShapeMismatchError,
)
from ._linalg import inv_a_norm, solve, valid
from ._signal import shrink2id, vol_adj

_MIN_CORR_DENOM: float = 1e-14  # guard against near-zero variance in correlation computation
_MAX_NAN_FRACTION: float = 0.9  # raise if more than this fraction of prices in any asset column are null

_logger = logging.getLogger(__name__)


def _ewm_corr_numpy(data: np.ndarray, com: int, min_periods: int) -> np.ndarray:
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
        result = np.where(denom > _MIN_CORR_DENOM, cov / denom, np.nan)

    result = np.clip(result, -1.0, 1.0)

    # Apply min_periods mask for all pairs
    result[count < min_periods] = np.nan

    # Diagonal is exactly 1.0 where the asset has sufficient observations
    diag_idx = np.arange(n_assets)
    diag_count = count[:, diag_idx, diag_idx]  # (T, N)
    result[:, diag_idx, diag_idx] = np.where(diag_count >= min_periods, 1.0, np.nan)

    return result


class BasanosConfig(BaseModel):
    r"""Configuration for correlation-aware position optimization.

    The required parameters (``vola``, ``corr``, ``clip``, ``shrink``, ``aum``)
    must be supplied by the caller.  The optional parameters carry
    carefully chosen defaults whose rationale is described below.

    Shrinkage methodology
    ---------------------
    ``shrink`` controls linear shrinkage of the EWMA correlation matrix toward
    the identity:

    .. math::

        C_{\\text{shrunk}} = \\lambda \\cdot C_{\\text{EWMA}} + (1 - \\lambda) \\cdot I_n

    where :math:`\\lambda` = ``shrink`` and :math:`I_n` is the identity.
    Shrinkage regularises the matrix when assets are few relative to the
    lookback (high concentration ratio :math:`n / T`), reducing the impact of
    extreme sample eigenvalues and improving the condition number of the matrix
    passed to the linear solver.

    **When to prefer strong shrinkage (low** ``shrink`` **/ high** ``1-shrink``\\ **):**

    * Fewer than ~30 assets with a ``corr`` lookback shorter than 100 days.
    * High-volatility or crisis regimes where correlations spike and the sample
      matrix is less representative of the true structure.
    * Portfolios where estimation noise is more costly than correlation bias
      (e.g., when the signal-to-noise ratio of ``mu`` is low).

    **When to prefer light shrinkage (high** ``shrink``\\ **):**

    * Many assets with a long lookback (low concentration ratio).
    * The EWMA correlation structure carries genuine diversification information
      that you want the solver to exploit.
    * Out-of-sample testing shows that position stability is not a concern.

    **Practical starting points (daily return data):**

    Here *n* = number of assets and *T* = ``cfg.corr`` (EWMA lookback).

    +-----------------------+-------------------+--------------------------------+
    | n (assets) / T (corr) | Suggested shrink  | Notes                          |
    +=======================+===================+================================+
    | n > 20, T < 40        | 0.3 - 0.5         | Near-singular matrix likely;   |
    |                       |                   | strong regularisation needed.  |
    +-----------------------+-------------------+--------------------------------+
    | n ~ 10, T ~ 60        | 0.5 - 0.7         | Balanced regime.               |
    +-----------------------+-------------------+--------------------------------+
    | n < 10, T > 100       | 0.7 - 0.9         | Well-conditioned sample;       |
    |                       |                   | light shrinkage for stability. |
    +-----------------------+-------------------+--------------------------------+

    See :func:`~basanos.math._signal.shrink2id` for the full theoretical
    background and academic references (Ledoit & Wolf, 2004; Chen et al., 2010).

    Default rationale
    -----------------
    ``profit_variance_init = 1.0``
        Unit variance is a neutral, uninformative starting point for the
        exponential moving average of realized P&L variance.  Because the
        normalised risk positions are divided by the square root of this
        quantity, initialising at 1.0 leaves the first few positions
        unaffected until the EMA accumulates real data.  Larger values
        would shrink initial positions; smaller values would inflate them.

    ``profit_variance_decay = 0.99``
        An EMA decay factor of λ = 0.99 corresponds to a half-life of
        ``log(0.5) / log(0.99) ≈ 69`` periods and an effective centre-of-
        mass of ``1 / (1 - 0.99) = 100`` periods.  For daily data this
        represents approximately 100 trading days (~5 months), a
        commonly used horizon for medium-frequency regime adaptation in
        systematic strategies.

    ``denom_tol = 1e-12``
        Positions are zeroed when the normalisation denominator
        ``inv_a_norm(μ, Σ)`` falls at or below this threshold.  The
        value 1e-12 provides ample headroom above float64 machine
        epsilon (~2.2e-16) while remaining negligible relative to any
        economically meaningful signal magnitude.

    ``position_scale = 1e6``
        The dimensionless risk position is multiplied by this factor
        before being passed to :class:`~basanos.analytics.Portfolio`.
        A value of 1e6 means positions are expressed in units of one
        million of the base currency, a conventional denomination for
        institutional-scale portfolios where AUM is measured in hundreds
        of millions.

    Examples:
        >>> cfg = BasanosConfig(vola=32, corr=64, clip=3.0, shrink=0.5, aum=1e8)
        >>> cfg.vola
        32
        >>> cfg.corr
        64
    """

    vola: int = Field(..., gt=0, description="EWMA lookback for volatility normalization.")
    corr: int = Field(..., gt=0, description="EWMA lookback for correlation estimation.")
    clip: float = Field(..., gt=0.0, description="Clipping threshold for volatility adjustment.")
    shrink: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Retention weight λ for linear shrinkage of the EWMA correlation matrix toward "
            "the identity: C_shrunk = λ·C_ewma + (1-λ)·I. "
            "λ=1.0 uses the raw EWMA matrix (no shrinkage); λ=0.0 replaces it entirely "
            "with the identity (maximum shrinkage, positions are treated as uncorrelated). "
            "Values in [0.3, 0.8] are typical for daily financial return data. "
            "Lower values improve numerical stability when assets are many relative to the "
            "lookback (high concentration ratio n/T). See shrink2id() for full guidance."
        ),
    )
    aum: float = Field(..., gt=0.0, description="Assets under management for portfolio scaling.")
    profit_variance_init: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Initial value for the profit variance EMA used in position sizing. "
            "Defaults to 1.0 (unit variance) so that the first positions are unscaled "
            "until real P&L data accumulates."
        ),
    )
    profit_variance_decay: float = Field(
        default=0.99,
        gt=0.0,
        lt=1.0,
        description=(
            "EMA decay factor λ for the realized P&L variance (higher = slower adaptation). "
            "The default 0.99 gives a half-life of ~69 periods and an effective window of "
            "100 periods, suitable for daily data."
        ),
    )
    denom_tol: float = Field(
        default=1e-12,
        gt=0.0,
        description=(
            "Minimum normalisation denominator; positions are zeroed at or below this value. "
            "The default 1e-12 is well above float64 machine epsilon (~2.2e-16) while "
            "remaining negligible for any economically meaningful signal."
        ),
    )
    position_scale: float = Field(
        default=1e6,
        gt=0.0,
        description=(
            "Multiplicative scaling factor applied to dimensionless risk positions to obtain "
            "cash positions in base-currency units. Defaults to 1e6 (one million), a "
            "conventional denomination for institutional portfolios."
        ),
    )

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("corr")
    @classmethod
    def corr_greater_than_vola(cls, v: int, info: ValidationInfo) -> int:
        """Optionally enforce corr ≥ vola for stability.

        Pydantic v2 passes ValidationInfo; use info.data to access other fields.
        """
        vola = info.data.get("vola") if hasattr(info, "data") else None
        if vola is not None and v < vola:
            raise ValueError
        return v


@dataclasses.dataclass(frozen=True)
class BasanosEngine:
    """Engine to compute correlation matrices and optimize risk positions.

    Encapsulates price data and configuration to build EWM-based
    correlations, apply shrinkage, and solve for normalized positions.
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

        # check for excessive NaN values: more than _MAX_NAN_FRACTION null in any asset column
        n_rows = self.prices.height
        if n_rows > 0:
            for asset in self.assets:
                nan_frac = self.prices[asset].null_count() / n_rows
                if nan_frac > _MAX_NAN_FRACTION:
                    raise ExcessiveNullsError(asset, nan_frac, _MAX_NAN_FRACTION)

        # check for monotonic price series: a strictly non-decreasing or non-increasing
        # series has no variance in its return sign, indicating malformed or synthetic data
        for asset in self.assets:
            col = self.prices[asset].drop_nulls()
            if col.len() > 2:
                diffs = col.diff().drop_nulls()
                if (diffs >= 0).all() or (diffs <= 0).all():
                    raise MonotonicPricesError(asset)

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
        tensor = _ewm_corr_numpy(ret_adj_np, com=self.cfg.corr, min_periods=self.cfg.corr)
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

    @property
    def cash_position(self) -> pl.DataFrame:
        """Optimize correlation-aware risk positions for each timestamp.

        Computes EWMA correlations (via ``self.cor``), applies shrinkage toward
        the identity matrix with intensity ``cfg.shrink``, and solves a
        normalized linear system A x = mu per timestamp to obtain stable,
        scale-invariant positions. Non-finite or ill-posed cases yield zero
        positions for safety.

        Returns:
            pl.DataFrame: DataFrame with columns ['date'] + asset names containing
            the per-timestamp cash positions (risk divided by EWMA volatility).

        Performance:
            Dominant cost is ``self.cor`` (O(T·N²) time, O(T·N²) memory — see
            :func:`_ewm_corr_numpy`).  The per-timestamp linear solve via
            Cholesky / LU decomposition adds O(N³) per row for a total solve
            cost of O(T·N³).  For *N* = 100 and *T* = 2 520 (~10 years daily)
            the solve contributes approximately 2.52 * 10^9 floating-point
            operations (upper bound; Cholesky is N³/3 + O(N²)); at *N* = 1 000
            this rises to roughly 2.52 * 10^12, making the per-row solve the
            dominant compute bottleneck for large universes.
        """
        # compute the correlation matrices
        cor = self.cor
        assets = self.assets

        # Compute risk positions row-by-row using correlation shrinkage (NumPy)
        prices_num = self.prices.select(assets).to_numpy()
        returns_num = np.zeros_like(prices_num, dtype=float)
        returns_num[1:] = prices_num[1:] / prices_num[:-1] - 1.0

        mu = self.mu.select(assets).to_numpy()
        risk_pos_np = np.full_like(mu, fill_value=np.nan, dtype=float)
        cash_pos_np = np.full_like(mu, fill_value=np.nan, dtype=float)
        vola_np = self.vola.select(assets).to_numpy()

        profit_variance = self.cfg.profit_variance_init
        lamb = self.cfg.profit_variance_decay

        for i, t in enumerate(cor.keys()):
            # get the mask of finite prices for this timestamp
            mask = np.isfinite(prices_num[i])

            # Compute profit contribution using only finite returns and available positions
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
            # we have no price data at all for this timestamp
            if not mask.any():
                continue

            # get the correlation matrix for this timestamp
            corr_n = cor[t]

            # shrink the correlation matrix towards identity
            matrix = shrink2id(corr_n, lamb=self.cfg.shrink)[np.ix_(mask, mask)]

            # get the expected-return vector for this timestamp
            expected_mu = np.nan_to_num(mu[i][mask])

            # Short-circuit when signal is zero - no position needed, skip norm computation
            if np.allclose(expected_mu, 0.0):
                pos = np.zeros_like(expected_mu)
            else:
                # Normalize solution; guard against zero/near-zero norm to avoid NaNs.
                # inv_a_norm returns float(np.nan) when no valid entries exist (never None).
                denom = inv_a_norm(expected_mu, matrix)

                if not np.isfinite(denom) or denom <= self.cfg.denom_tol:
                    _logger.warning(
                        "Positions zeroed at t=%s: normalisation denominator is degenerate "
                        "(denom=%s, denom_tol=%s). Check signal magnitude and covariance matrix.",
                        t,
                        denom,
                        self.cfg.denom_tol,
                    )
                    pos = np.zeros_like(expected_mu)
                else:
                    pos = solve(matrix, expected_mu) / denom

            risk_pos_np[i, mask] = pos / profit_variance
            with np.errstate(invalid="ignore"):
                cash_pos_np[i, mask] = risk_pos_np[i, mask] / vola_np[i, mask]

        # Build Polars DataFrame for risk positions (numeric columns only)
        cash_position = self.prices.with_columns(
            [(pl.lit(cash_pos_np[:, i]).alias(asset)) for i, asset in enumerate(assets)]
        )

        return cash_position

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

    @property
    def condition_number(self) -> pl.DataFrame:
        """Condition number κ of the shrunk correlation matrix C_shrunk at each timestamp.

        Applies the same shrinkage as ``cash_position`` (``cfg.shrink``) to the
        per-timestamp EWM correlation matrix and computes ``np.linalg.cond``.
        Only the sub-matrix corresponding to assets with finite prices at that
        timestamp is used; rows with no finite prices yield ``NaN``.

        Returns:
            pl.DataFrame: Two-column DataFrame ``{'date': ..., 'condition_number': ...}``.
        """
        cor = self.cor
        assets = self.assets
        prices_num = self.prices.select(assets).to_numpy()

        kappas: list[float] = []
        for i, (_t, corr_n) in enumerate(cor.items()):
            mask = np.isfinite(prices_num[i])
            if not mask.any():
                kappas.append(float(np.nan))
                continue
            matrix = shrink2id(corr_n, lamb=self.cfg.shrink)[np.ix_(mask, mask)]
            _v, mat = valid(matrix)
            if not _v.any():
                kappas.append(float(np.nan))
                continue
            kappas.append(float(np.linalg.cond(mat)))

        return pl.DataFrame({"date": self.prices["date"], "condition_number": pl.Series(kappas, dtype=pl.Float64)})

    @property
    def effective_rank(self) -> pl.DataFrame:
        r"""Effective rank of the shrunk correlation matrix C_shrunk at each timestamp.

        Measures the true dimensionality of the portfolio by computing the
        entropy-based effective rank:

        .. math::

            \\text{eff\\_rank} = \\exp\\!\\left(-\\sum_i p_i \\ln p_i\\right),
            \\quad p_i = \\frac{\\lambda_i}{\\sum_j \\lambda_j}

        where :math:`\\lambda_i` are the eigenvalues of ``C_shrunk`` (restricted
        to assets with finite prices at that timestamp).  A value equal to the
        number of assets indicates a perfectly uniform spectrum (identity-like
        matrix); a value of 1 indicates a rank-1 matrix dominated by one
        direction.

        Returns:
            pl.DataFrame: Two-column DataFrame ``{'date': ..., 'effective_rank': ...}``.
        """
        cor = self.cor
        assets = self.assets
        prices_num = self.prices.select(assets).to_numpy()

        ranks: list[float] = []
        for i, (_t, corr_n) in enumerate(cor.items()):
            mask = np.isfinite(prices_num[i])
            if not mask.any():
                ranks.append(float(np.nan))
                continue
            matrix = shrink2id(corr_n, lamb=self.cfg.shrink)[np.ix_(mask, mask)]
            _v, mat = valid(matrix)
            if not _v.any():
                ranks.append(float(np.nan))
                continue
            eigvals = np.linalg.eigvalsh(mat)
            eigvals = np.clip(eigvals, 0.0, None)
            total = eigvals.sum()
            if total <= 0.0:
                ranks.append(float(np.nan))
                continue
            p = eigvals / total
            p_pos = p[p > 0.0]
            entropy = float(-np.sum(p_pos * np.log(p_pos)))
            ranks.append(float(np.exp(entropy)))

        return pl.DataFrame({"date": self.prices["date"], "effective_rank": pl.Series(ranks, dtype=pl.Float64)})

    @property
    def solver_residual(self) -> pl.DataFrame:
        r"""Per-timestamp solver residual ``‖C_shrunk·x - μ‖₂``.

        After solving the normalised linear system ``C_shrunk · x = μ`` at
        each timestamp, this property reports the Euclidean residual norm.
        For a well-posed, well-conditioned system the residual is near machine
        epsilon; large values flag numerical difficulties (near-singular
        matrices, extreme condition numbers, or solver fall-back to LU).

        Returns:
            pl.DataFrame: Two-column DataFrame ``{'date': ..., 'residual': ...}``.
            Zero is returned when ``μ`` is the zero vector (no solve is
            performed).  ``NaN`` is returned when no asset has finite prices.
        """
        cor = self.cor
        assets = self.assets
        mu_np = self.mu.select(assets).to_numpy()
        prices_num = self.prices.select(assets).to_numpy()

        residuals: list[float] = []
        for i, (_t, corr_n) in enumerate(cor.items()):
            mask = np.isfinite(prices_num[i])
            if not mask.any():
                residuals.append(float(np.nan))
                continue
            matrix = shrink2id(corr_n, lamb=self.cfg.shrink)[np.ix_(mask, mask)]
            expected_mu = np.nan_to_num(mu_np[i][mask])
            if np.allclose(expected_mu, 0.0):
                residuals.append(0.0)
                continue
            x = solve(matrix, expected_mu)
            finite_x = np.isfinite(x)
            if not finite_x.any():
                residuals.append(float(np.nan))
                continue
            residuals.append(
                float(np.linalg.norm(matrix[np.ix_(finite_x, finite_x)] @ x[finite_x] - expected_mu[finite_x]))
            )

        return pl.DataFrame({"date": self.prices["date"], "residual": pl.Series(residuals, dtype=pl.Float64)})

    @property
    def signal_utilisation(self) -> pl.DataFrame:
        r"""Per-asset signal utilisation: fraction of μ_i surviving the correlation filter.

        For each asset *i* and timestamp *t*, computes

        .. math::

            u_i = \\frac{(C_{\\text{shrunk}}^{-1}\\,\\mu)_i}{\\mu_i}

        where :math:`C_{\\text{shrunk}}^{-1}\\,\\mu` is the unnormalised solve
        result.  When :math:`C = I` (identity) all assets have utilisation 1.
        Off-diagonal correlations attenuate some assets (:math:`u_i < 1`) and
        may amplify negatively correlated ones (:math:`u_i > 1`).

        A value of ``0.0`` is returned when the entire signal vector
        :math:`\\mu` is near zero at that timestamp (no solve is performed).
        ``NaN`` is returned for individual assets where :math:`|\\mu_i|` is
        below machine-epsilon precision or where prices are unavailable.

        Returns:
            pl.DataFrame: DataFrame with columns ``['date'] + assets``.
        """
        cor = self.cor
        assets = self.assets
        mu_np = self.mu.select(assets).to_numpy()
        prices_num = self.prices.select(assets).to_numpy()

        _mu_tol = 1e-14  # treat |μ_i| below this as zero to avoid spurious large ratios
        n_assets = len(assets)
        util_np = np.full((self.prices.height, n_assets), np.nan)

        for i, (_t, corr_n) in enumerate(cor.items()):
            mask = np.isfinite(prices_num[i])
            if not mask.any():
                continue
            matrix = shrink2id(corr_n, lamb=self.cfg.shrink)[np.ix_(mask, mask)]
            expected_mu = np.nan_to_num(mu_np[i][mask])
            if np.allclose(expected_mu, 0.0):
                util_np[i, mask] = 0.0
                continue
            x = solve(matrix, expected_mu)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(np.abs(expected_mu) > _mu_tol, x / expected_mu, np.nan)
            util_np[i, mask] = ratio

        return self.prices.with_columns([pl.lit(util_np[:, j]).alias(asset) for j, asset in enumerate(assets)])

    @property
    def portfolio(self) -> Portfolio:
        """Construct a Portfolio from the optimized cash positions.

        Converts the computed cash positions into a Portfolio using the
        configured AUM.

        Returns:
            Portfolio: Instance built from cash positions with AUM scaling.
        """
<<<<<<< copilot/implement-turnover-measures
        cp = self.cash_position
        assets = [c for c in cp.columns if c != "date" and cp[c].dtype.is_numeric()]
        scaled = cp.with_columns(pl.col(a) * self.cfg.position_scale for a in assets)
        return Portfolio.from_cash_position(self.prices, scaled, aum=self.cfg.aum)
=======
        return Portfolio.from_cash_position(self.prices, self.cash_position * self.cfg.position_scale, aum=self.cfg.aum)

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
        new_cfg = self.cfg.model_copy(update={"shrink": shrink})
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

    def _ic_series(self, use_rank: bool) -> pl.DataFrame:
        """Compute the cross-sectional IC time series.

        For each timestamp *t* (from 0 to T-2), correlates the signal vector
        ``mu[t, :]`` with the one-period forward return vector
        ``prices[t+1, :] / prices[t, :] - 1`` across all assets where both
        quantities are finite.  When fewer than two valid asset pairs are
        available, the IC value is set to ``NaN``.

        Args:
            use_rank: When ``True`` the Spearman rank correlation is used
                (Rank IC); when ``False`` the Pearson correlation is used (IC).

        Returns:
            pl.DataFrame: Two-column frame with ``date`` (signal date) and
            either ``ic`` or ``rank_ic``.
        """
        assets = self.assets
        prices_np = self.prices.select(assets).to_numpy().astype(float)
        mu_np = self.mu.select(assets).to_numpy().astype(float)
        dates = self.prices["date"].to_list()

        col_name = "rank_ic" if use_rank else "ic"
        ic_values: list[float] = []
        ic_dates = []

        for t in range(len(dates) - 1):
            fwd_ret = prices_np[t + 1] / prices_np[t] - 1.0
            signal = mu_np[t]

            # Both signal and forward return must be finite
            mask = np.isfinite(signal) & np.isfinite(fwd_ret)
            n_valid = int(mask.sum())

            if n_valid < 2:
                ic_values.append(float("nan"))
            elif use_rank:
                corr, _ = spearmanr(signal[mask], fwd_ret[mask])
                ic_values.append(float(corr))
            else:
                ic_values.append(float(np.corrcoef(signal[mask], fwd_ret[mask])[0, 1]))

            ic_dates.append(dates[t])

        return pl.DataFrame({"date": ic_dates, col_name: pl.Series(ic_values, dtype=pl.Float64)})

    @property
    def ic(self) -> pl.DataFrame:
        """Cross-sectional Pearson Information Coefficient (IC) time series.

        For each timestamp *t* (excluding the last), computes the Pearson
        correlation between the signal ``mu[t, :]`` and the one-period forward
        return ``prices[t+1, :] / prices[t, :] - 1`` across all assets where
        both quantities are finite.

        An IC value close to +1 means the signal ranked assets in the same
        order as forward returns; close to -1 means the opposite; near 0 means
        no predictive relationship.

        Returns:
            pl.DataFrame: Frame with columns ``['date', 'ic']``.  ``date`` is
            the timestamp at which the signal was observed.  ``ic`` is a
            ``Float64`` series (``NaN`` when fewer than 2 valid asset pairs
            are available for a given timestamp).

        See Also:
            :py:attr:`rank_ic` — Spearman variant, more robust to outliers.
            :py:attr:`ic_mean`, :py:attr:`ic_std`, :py:attr:`icir` — summary
            statistics.
        """
        return self._ic_series(use_rank=False)

    @property
    def rank_ic(self) -> pl.DataFrame:
        """Cross-sectional Spearman Rank Information Coefficient time series.

        Identical to :py:attr:`ic` but uses the Spearman rank correlation
        instead of the Pearson correlation, making it more robust to fat-tailed
        return distributions and outliers.

        Returns:
            pl.DataFrame: Frame with columns ``['date', 'rank_ic']``.
            ``rank_ic`` is a ``Float64`` series.

        See Also:
            :py:attr:`ic` — Pearson variant.
            :py:attr:`rank_ic_mean`, :py:attr:`rank_ic_std` — summary
            statistics.
        """
        return self._ic_series(use_rank=True)

    @property
    def ic_mean(self) -> float:
        """Mean of the IC time series, ignoring NaN values.

        Returns:
            float: Arithmetic mean of all finite IC values, or ``NaN`` if
            no finite values exist.
        """
        arr = self.ic["ic"].drop_nulls().to_numpy()
        finite = arr[np.isfinite(arr)]
        return float(np.mean(finite)) if len(finite) > 0 else float("nan")

    @property
    def ic_std(self) -> float:
        """Standard deviation of the IC time series, ignoring NaN values.

        Uses ``ddof=1`` (sample standard deviation).

        Returns:
            float: Sample standard deviation of all finite IC values, or
            ``NaN`` if fewer than 2 finite values exist.
        """
        arr = self.ic["ic"].drop_nulls().to_numpy()
        finite = arr[np.isfinite(arr)]
        return float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")

    @property
    def icir(self) -> float:
        """Information Coefficient Information Ratio (ICIR).

        Defined as ``IC mean / IC std``.  A higher absolute ICIR indicates a
        more consistent signal: the mean IC is large relative to its
        variability.

        Returns:
            float: ``ic_mean / ic_std``, or ``NaN`` when ``ic_std`` is zero
            or non-finite.
        """
        mean = self.ic_mean
        std = self.ic_std
        if not np.isfinite(std) or std == 0.0:
            return float("nan")
        return float(mean / std)

    @property
    def rank_ic_mean(self) -> float:
        """Mean of the Rank IC time series, ignoring NaN values.

        Returns:
            float: Arithmetic mean of all finite Rank IC values, or ``NaN``
            if no finite values exist.
        """
        arr = self.rank_ic["rank_ic"].drop_nulls().to_numpy()
        finite = arr[np.isfinite(arr)]
        return float(np.mean(finite)) if len(finite) > 0 else float("nan")

    @property
    def rank_ic_std(self) -> float:
        """Standard deviation of the Rank IC time series, ignoring NaN values.

        Uses ``ddof=1`` (sample standard deviation).

        Returns:
            float: Sample standard deviation of all finite Rank IC values, or
            ``NaN`` if fewer than 2 finite values exist.
        """
        arr = self.rank_ic["rank_ic"].drop_nulls().to_numpy()
        finite = arr[np.isfinite(arr)]
        return float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
>>>>>>> main
