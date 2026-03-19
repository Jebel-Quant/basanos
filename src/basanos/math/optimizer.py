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
import enum
import logging
from typing import TYPE_CHECKING, Annotated, Literal, cast

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
    SingularMatrixError,
)
from ._factor_model import FactorModel
from ._linalg import inv_a_norm, solve, valid
from ._signal import shrink2id, vol_adj

if TYPE_CHECKING:
    from ._config_report import ConfigReport

_logger = logging.getLogger(__name__)


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


class CovarianceMode(enum.StrEnum):
    r"""Covariance estimation mode for the Basanos optimizer.

    Attributes:
        ewma_shrink: EWMA correlation matrix with linear shrinkage toward the
            identity.  Controlled by :attr:`BasanosConfig.shrink`.
            This is the default mode.
        sliding_window: Rolling-window factor model.  A fixed block of the
            ``W`` most recent volatility-adjusted returns is decomposed via
            truncated SVD into ``k`` latent factors, giving the estimator

            .. math::

                \\hat{C}_t^{(W,k)} = \\frac{1}{W}
                    \\mathbf{V}_{k,t}\\mathbf{\\Sigma}_{k,t}^2\\mathbf{V}_{k,t}^\\top
                    + \\hat{D}_t

            where :math:`\\hat{D}_t` is chosen to enforce unit diagonal.
            The system is solved efficiently via the Woodbury identity
            (Section 4.3 of basanos.pdf) at :math:`O(k^3 + kn)` per step
            rather than :math:`O(n^3)`.
            Configured via :class:`SlidingWindowConfig`.

    Examples:
        >>> CovarianceMode.ewma_shrink
        <CovarianceMode.ewma_shrink: 'ewma_shrink'>
        >>> CovarianceMode.sliding_window
        <CovarianceMode.sliding_window: 'sliding_window'>
        >>> CovarianceMode("sliding_window")
        <CovarianceMode.sliding_window: 'sliding_window'>
    """

    ewma_shrink = "ewma_shrink"
    sliding_window = "sliding_window"


class EwmaShrinkConfig(BaseModel):
    """Covariance configuration for the ``ewma_shrink`` mode.

    This is the default covariance mode. No additional parameters are required
    beyond those already present on :class:`BasanosConfig` (``shrink``, ``corr``).

    Examples:
        >>> cfg = EwmaShrinkConfig()
        >>> cfg.covariance_mode
        <CovarianceMode.ewma_shrink: 'ewma_shrink'>
    """

    covariance_mode: Literal[CovarianceMode.ewma_shrink] = CovarianceMode.ewma_shrink

    model_config = {"frozen": True}


class SlidingWindowConfig(BaseModel):
    r"""Covariance configuration for the ``sliding_window`` mode.

    Requires both ``window`` (rolling window length) and ``n_factors`` (number
    of latent factors for the truncated SVD factor model).

    Args:
        window: Rolling window length :math:`W \\geq 1`.
            Rule of thumb: :math:`W \\geq 2n` keeps the sample covariance
            well-posed before truncation.
        n_factors: Number of latent factors :math:`k \\geq 1`.
            :math:`k = 1` recovers the single market-factor model; larger
            :math:`k` captures finer correlation structure at the cost of
            higher estimation noise.

    Examples:
        >>> cfg = SlidingWindowConfig(window=60, n_factors=3)
        >>> cfg.covariance_mode
        <CovarianceMode.sliding_window: 'sliding_window'>
        >>> cfg.window
        60
        >>> cfg.n_factors
        3
    """

    covariance_mode: Literal[CovarianceMode.sliding_window] = CovarianceMode.sliding_window
    window: int = Field(
        ...,
        gt=0,
        description=(
            "Sliding window length W (number of most recent observations). "
            "Rule of thumb: W >= 2 * n_assets to keep the sample covariance well-posed."
        ),
    )
    n_factors: int = Field(
        ...,
        gt=0,
        description=(
            "Number of latent factors k for the sliding window factor model. "
            "k=1 recovers the single market-factor model; larger k captures finer correlation "
            "structure at the cost of higher estimation noise."
        ),
    )

    model_config = {"frozen": True}


CovarianceConfig = Annotated[
    EwmaShrinkConfig | SlidingWindowConfig,
    Field(discriminator="covariance_mode"),
]
"""Discriminated union of covariance-mode configurations.

Pydantic selects the correct sub-config based on the ``covariance_mode``
discriminator field:

* :class:`EwmaShrinkConfig` when ``covariance_mode="ewma_shrink"``
* :class:`SlidingWindowConfig` when ``covariance_mode="sliding_window"``
"""


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

    ``min_corr_denom = 1e-14``
        The EWMA correlation denominator ``sqrt(var_x * var_y)`` is
        compared against this threshold; when at or below it the
        correlation is set to NaN rather than dividing by a near-zero
        value.  The default 1e-14 is safely above float64 underflow
        while remaining negligible for any realistic return series.
        Advanced users may tighten this guard (larger value) when
        working with very-low-variance synthetic data.

    ``max_nan_fraction = 0.9``
        :class:`~basanos.exceptions.ExcessiveNullsError` is raised
        during construction when the null fraction in any asset price
        column **strictly exceeds** this threshold.  The default 0.9
        permits up to 90 % missing prices (e.g., illiquid or recently
        listed assets in a long history) while rejecting columns that
        are almost entirely null and would contribute no useful
        information.  Callers who want a stricter gate can lower this
        value; callers running on sparse data can raise it toward 1.0.

    Sliding-window mode
    -------------------
    When ``covariance_config`` is a :class:`SlidingWindowConfig`, the EWMA
    correlation estimator is replaced by a rolling-window factor model
    (Section 4.4 of basanos.pdf).  At each timestamp *t* the
    :math:`W \\times n` submatrix of the :math:`W` most recent
    volatility-adjusted returns is decomposed via truncated SVD to extract
    :math:`k` latent factors.  The resulting correlation estimate is

    .. math::

        \\hat{C}_t^{(W,k)}
        = \\frac{1}{W}\\mathbf{V}_{k,t}\\mathbf{\\Sigma}_{k,t}^2
          \\mathbf{V}_{k,t}^\\top + \\hat{D}_t

    where :math:`\\hat{D}_t` enforces unit diagonal.  The linear system
    :math:`\\hat{C}_t^{(W,k)}\\mathbf{x}_t = \\boldsymbol{\\mu}_t` is solved
    via the Woodbury identity (:func:`~basanos.math._factor_model.FactorModel.solve`)
    at cost :math:`O(k^3 + kn)` per step rather than :math:`O(n^3)`.

    ``covariance_config``
        Pass a :class:`SlidingWindowConfig` instance to enable this mode.
        The required sub-parameters are:

        ``window``
            Rolling window length :math:`W \\geq 1`.  Rule of thumb: :math:`W
            \\geq 2n` keeps the sample covariance well-posed before truncation.

        ``n_factors``
            Number of latent factors :math:`k \\geq 1`.  :math:`k = 1`
            recovers the single market-factor model; larger :math:`k` captures
            finer correlation structure at the cost of higher estimation noise.

    Examples:
        >>> cfg = BasanosConfig(vola=32, corr=64, clip=3.0, shrink=0.5, aum=1e8)
        >>> cfg.vola
        32
        >>> cfg.corr
        64
        >>> sw_cfg = BasanosConfig(
        ...     vola=16, corr=32, clip=3.0, shrink=0.5, aum=1e6,
        ...     covariance_config=SlidingWindowConfig(window=60, n_factors=3),
        ... )
        >>> sw_cfg.covariance_mode
        <CovarianceMode.sliding_window: 'sliding_window'>
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
            "lookback (high concentration ratio n/T). See shrink2id() for full guidance. "
            "Only used when covariance_mode='ewma_shrink'."
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
    min_corr_denom: float = Field(
        default=1e-14,
        gt=0.0,
        description=(
            "Guard threshold for the EWMA correlation denominator sqrt(var_x * var_y). "
            "When the denominator is at or below this value the correlation is set to NaN "
            "instead of dividing by a near-zero number. "
            "The default 1e-14 is safely above float64 underflow while being negligible for "
            "any realistic return variance."
        ),
    )
    max_nan_fraction: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        description=(
            "Maximum tolerated fraction of null values in any asset price column. "
            "ExcessiveNullsError is raised during construction when the null fraction "
            "strictly exceeds this threshold. "
            "The default 0.9 allows up to 90 % missing prices while rejecting columns "
            "that are almost entirely null."
        ),
    )
    covariance_config: CovarianceConfig = Field(
        default_factory=EwmaShrinkConfig,
        description=(
            "Covariance estimation configuration. "
            "Pass EwmaShrinkConfig() (default) for EWMA correlation with linear shrinkage "
            "toward the identity, or SlidingWindowConfig(window=W, n_factors=k) for a "
            "rolling-window factor model. See Section 4.4 of basanos.pdf."
        ),
    )

    model_config = {"frozen": True, "extra": "forbid"}

    @property
    def covariance_mode(self) -> CovarianceMode:
        """Covariance mode derived from :attr:`covariance_config`."""
        return self.covariance_config.covariance_mode

    @property
    def window(self) -> int | None:
        """Sliding window length, or ``None`` when not in ``sliding_window`` mode."""
        if isinstance(self.covariance_config, SlidingWindowConfig):
            return self.covariance_config.window
        return None

    @property
    def n_factors(self) -> int | None:
        """Number of latent factors, or ``None`` when not in ``sliding_window`` mode."""
        if isinstance(self.covariance_config, SlidingWindowConfig):
            return self.covariance_config.n_factors
        return None

    @property
    def report(self) -> "ConfigReport":
        """Return a :class:`~basanos.math._config_report.ConfigReport` facade for this config.

        Generates a self-contained HTML report summarising all configuration
        parameters, a shrinkage-guidance table, and a theory section on
        Ledoit-Wolf shrinkage.

        To also include a lambda-sweep chart (Sharpe vs λ), use
        :attr:`BasanosEngine.config_report` instead, which requires price and
        signal data.

        Returns:
            basanos.math._config_report.ConfigReport: Report facade with
            ``to_html()`` and ``save()`` methods.

        Examples:
            >>> from basanos.math.optimizer import BasanosConfig
            >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
            >>> report = cfg.report
            >>> html = report.to_html()
            >>> "Parameters" in html
            True
        """
        from ._config_report import ConfigReport

        return ConfigReport(config=self)

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

    def _iter_matrices(self):
        r"""Yield ``(i, t, mask, matrix)`` for every timestamp.

        ``matrix`` is the effective :math:`(n_{\\text{sub}},\\ n_{\\text{sub}})`
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
                if k_eff < 1:
                    yield i, t, mask, None
                    continue
                try:
                    fm = FactorModel.from_returns(window_ret, k=k_eff)
                    yield i, t, mask, fm.covariance
                except Exception:
                    yield i, t, mask, None

    @property
    def cash_position(self) -> pl.DataFrame:
        r"""Optimize correlation-aware risk positions for each timestamp.

        Supports two covariance modes controlled by ``cfg.covariance_config``:

        * :class:`EwmaShrinkConfig` (default): Computes EWMA correlations, applies
          linear shrinkage toward the identity, and solves a normalised linear
          system :math:`C\\,x = \\mu` per timestamp via Cholesky / LU.

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

        if isinstance(self.cfg.covariance_config, EwmaShrinkConfig):
            # ── EWMA / shrinkage path ───────────────────────────────────────
            cor = self.cor
            index_iter = list(cor.keys())
        else:
            # ── Sliding window path ─────────────────────────────────────────
            sw_config = cast(SlidingWindowConfig, self.cfg.covariance_config)
            cor = None
            index_iter = self.prices["date"].to_list()
            ret_adj_np = self.ret_adj.select(assets).to_numpy()
            win_w: int = sw_config.window
            win_k: int = sw_config.n_factors

        for i, t in enumerate(index_iter):
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

            if isinstance(self.cfg.covariance_config, EwmaShrinkConfig):
                # get the correlation matrix for this timestamp and shrink it
                corr_n = cor[t]  # type: ignore[index]
                matrix = shrink2id(corr_n, lamb=self.cfg.shrink)[np.ix_(mask, mask)]

                # get the expected-return vector for this timestamp
                expected_mu = np.nan_to_num(mu[i][mask])

                # Short-circuit when signal is zero
                if np.allclose(expected_mu, 0.0):
                    pos = np.zeros_like(expected_mu)
                else:
                    denom = inv_a_norm(expected_mu, matrix)
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
                        pos = np.zeros_like(expected_mu)
                    else:
                        pos = solve(matrix, expected_mu) / denom
            else:
                # ── Sliding window: fit factor model on the last W rows ─────
                if i + 1 < win_w:
                    continue  # not enough history yet

                # Extract the (W, n_sub) window of vol-adjusted returns
                window_ret = ret_adj_np[i + 1 - win_w : i + 1][:, mask]
                # Replace NaN with 0 (neutral: no systematic contribution)
                window_ret = np.where(np.isfinite(window_ret), window_ret, 0.0)

                n_sub = int(mask.sum())
                k_eff = min(win_k, win_w, n_sub)
                if k_eff < 1:
                    continue

                try:
                    fm = FactorModel.from_returns(window_ret, k=k_eff)
                except Exception as exc:
                    _logger.debug("Sliding window SVD failed at t=%s: %s", t, exc)
                    continue

                expected_mu = np.nan_to_num(mu[i][mask])

                if np.allclose(expected_mu, 0.0):
                    pos = np.zeros(n_sub)
                else:
                    try:
                        # Solve Ĉ x = μ via Woodbury identity: x = fm.solve(μ)
                        x = fm.solve(expected_mu)
                        # Normalisation denominator sqrt(μᵀ C⁻¹ μ) = sqrt(μᵀ x)
                        denom = float(np.sqrt(max(0.0, float(np.dot(expected_mu, x)))))
                    except Exception:
                        pos = np.zeros(n_sub)
                    else:
                        if not np.isfinite(denom) or denom <= self.cfg.denom_tol:
                            _logger.warning(
                                "Positions zeroed at t=%s (sliding_window): normalisation "
                                "denominator is degenerate (denom=%s, denom_tol=%s).",
                                t,
                                denom,
                                self.cfg.denom_tol,
                            )
                            pos = np.zeros(n_sub)
                        else:
                            pos = x / denom

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
        """Condition number κ of the effective correlation matrix at each timestamp.

        Uses the same covariance mode as :attr:`cash_position`: for
        ``ewma_shrink`` this is the shrunk EWMA matrix; for ``sliding_window``
        it is the factor-model covariance.  Only the sub-matrix corresponding
        to assets with finite prices at that timestamp is used; rows with no
        finite prices yield ``NaN``.

        Returns:
            pl.DataFrame: Two-column DataFrame ``{'date': ..., 'condition_number': ...}``.
        """
        kappas: list[float] = []
        for _i, _t, _mask, matrix in self._iter_matrices():
            if matrix is None:
                kappas.append(float(np.nan))
                continue
            _v, mat = valid(matrix)
            if not _v.any():
                kappas.append(float(np.nan))
                continue
            kappas.append(float(np.linalg.cond(mat)))

        return pl.DataFrame({"date": self.prices["date"], "condition_number": pl.Series(kappas, dtype=pl.Float64)})

    @property
    def effective_rank(self) -> pl.DataFrame:
        r"""Effective rank of the effective correlation matrix at each timestamp.

        Measures the true dimensionality of the portfolio by computing the
        entropy-based effective rank:

        .. math::

            \\text{eff\\_rank} = \\exp\\!\\left(-\\sum_i p_i \\ln p_i\\right),
            \\quad p_i = \\frac{\\lambda_i}{\\sum_j \\lambda_j}

        where :math:`\\lambda_i` are the eigenvalues of the effective
        correlation matrix (restricted to assets with finite prices at that
        timestamp).  Uses the same covariance mode as :attr:`cash_position`.
        A value equal to the number of assets indicates a perfectly uniform
        spectrum; a value of 1 indicates a rank-1 matrix.

        Returns:
            pl.DataFrame: Two-column DataFrame ``{'date': ..., 'effective_rank': ...}``.
        """
        ranks: list[float] = []
        for _i, _t, _mask, matrix in self._iter_matrices():
            if matrix is None:
                ranks.append(float(np.nan))
                continue
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
        r"""Per-timestamp solver residual ``‖C·x - μ‖₂``.

        After solving the normalised linear system ``C · x = μ`` at
        each timestamp, this property reports the Euclidean residual norm.
        For a well-posed, well-conditioned system the residual is near machine
        epsilon; large values flag numerical difficulties (near-singular
        matrices, extreme condition numbers, or solver fall-back to LU).
        Uses the same covariance mode as :attr:`cash_position`.

        Returns:
            pl.DataFrame: Two-column DataFrame ``{'date': ..., 'residual': ...}``.
            Zero is returned when ``μ`` is the zero vector (no solve is
            performed).  ``NaN`` is returned when no asset has finite prices.
        """
        assets = self.assets
        mu_np = self.mu.select(assets).to_numpy()

        residuals: list[float] = []
        for i, t, mask, matrix in self._iter_matrices():
            if matrix is None:
                residuals.append(float(np.nan))
                continue
            expected_mu = np.nan_to_num(mu_np[i][mask])
            if np.allclose(expected_mu, 0.0):
                residuals.append(0.0)
                continue
            try:
                x = solve(matrix, expected_mu)
            except SingularMatrixError:
                # The covariance matrix is degenerate — residual is undefined.
                _logger.warning(
                    "solver_residual: SingularMatrixError at t=%s - covariance matrix is "
                    "degenerate; residual set to NaN.",
                    t,
                )
                residuals.append(float(np.nan))
                continue
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

            u_i = \\frac{(C^{-1}\\,\\mu)_i}{\\mu_i}

        where :math:`C^{-1}\\,\\mu` is the unnormalised solve result using
        the effective correlation matrix for the current
        :attr:`~BasanosConfig.covariance_mode`.  When :math:`C = I`
        (identity) all assets have utilisation 1.  Off-diagonal correlations
        attenuate some assets (:math:`u_i < 1`) and may amplify negatively
        correlated ones (:math:`u_i > 1`).

        A value of ``0.0`` is returned when the entire signal vector
        :math:`\\mu` is near zero at that timestamp (no solve is performed).
        ``NaN`` is returned for individual assets where :math:`|\\mu_i|` is
        below machine-epsilon precision or where prices are unavailable.

        Returns:
            pl.DataFrame: DataFrame with columns ``['date'] + assets``.
        """
        assets = self.assets
        mu_np = self.mu.select(assets).to_numpy()

        _mu_tol = 1e-14  # treat |μ_i| below this as zero to avoid spurious large ratios
        n_assets = len(assets)
        util_np = np.full((self.prices.height, n_assets), np.nan)

        for i, t, mask, matrix in self._iter_matrices():
            if matrix is None:
                continue
            expected_mu = np.nan_to_num(mu_np[i][mask])
            if np.allclose(expected_mu, 0.0):
                util_np[i, mask] = 0.0
                continue
            try:
                x = solve(matrix, expected_mu)
            except SingularMatrixError:
                # The covariance matrix is degenerate — utilisation is undefined.
                _logger.warning(
                    "signal_utilisation: SingularMatrixError at t=%s - covariance matrix is "
                    "degenerate; utilisation set to NaN.",
                    t,
                )
                continue
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
        new_cfg = self.cfg.model_copy(update={"shrink": shrink})
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
            window: Rolling window length :math:`W \\geq 1`.
                Rule of thumb: :math:`W \\geq 2 \\cdot n_{\\text{assets}}`.
            n_factors: Number of latent factors :math:`k \\geq 1`.

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
        new_cfg = self.cfg.model_copy(
            update={
                "covariance_config": SlidingWindowConfig(window=window, n_factors=n_factors),
            }
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
