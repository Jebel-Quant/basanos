"""Configuration classes for the Basanos optimizer.

Extracted from ``optimizer.py`` to keep each module focused on a single concern.
All public names are re-exported from ``optimizer.py`` so existing imports are
unaffected.
"""

import enum
import logging
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

if TYPE_CHECKING:
    from ._config_report import ConfigReport

_logger = logging.getLogger(__name__)


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

    .. note::
        This class is **intentionally minimal**. The only field is the
        ``covariance_mode`` discriminator, which is required to make Pydantic's
        discriminated-union dispatch work correctly (see :data:`CovarianceConfig`).
        Before adding new EWMA-specific fields here, consider whether the field
        name clashes with existing :class:`BasanosConfig` top-level fields and
        whether it would constitute a breaking change to the public API.

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

    **Effective component count** — at each streaming step the number of SVD
    components actually used is

    .. math::

        k_{\text{eff}} = \min(k,\; W,\; n_{\text{valid}},\; k_{\text{max}})

    where :math:`k` = ``n_factors``, :math:`W` = ``window``,
    :math:`n_{\text{valid}}` is the number of assets with finite prices at that
    step, and :math:`k_{\text{max}}` = ``max_components`` (or :math:`+\infty`
    when unset).  This ensures the truncated SVD remains well-posed even when
    assets temporarily drop out of the universe.  Setting ``max_components``
    explicitly caps computational cost in large universes without changing the
    desired factor count used in batch mode.

    Args:
        window: Rolling window length :math:`W \\geq 1`.
            Rule of thumb: :math:`W \\geq 2n` keeps the sample covariance
            well-posed before truncation.
        n_factors: Number of latent factors :math:`k \\geq 1`.
            :math:`k = 1` recovers the single market-factor model; larger
            :math:`k` captures finer correlation structure at the cost of
            higher estimation noise.
        max_components: Optional hard cap on the number of SVD components used
            per streaming step.  When set, the effective component count is
            :math:`\\min(k_{\\text{eff}},\\, \\texttt{max\\_components})`.
            Useful for large universes where only a few factors dominate and
            you want to limit SVD cost below ``n_factors``.  Must be
            :math:`\\geq 1` when provided.  Defaults to ``None`` (no extra cap).

    Examples:
        >>> cfg = SlidingWindowConfig(window=60, n_factors=3)
        >>> cfg.covariance_mode
        <CovarianceMode.sliding_window: 'sliding_window'>
        >>> cfg.window
        60
        >>> cfg.n_factors
        3
        >>> cfg.max_components is None
        True
        >>> cfg2 = SlidingWindowConfig(window=60, n_factors=10, max_components=3)
        >>> cfg2.max_components
        3
    """

    covariance_mode: Literal[CovarianceMode.sliding_window] = CovarianceMode.sliding_window
    window: int = Field(
        ...,
        gt=0,
        description=(
            "Sliding window length W (number of most recent observations). "
            "Rule of thumb: W >= 2 * n_assets to keep the sample covariance well-posed. "
            "Note: the first W-1 rows of output will have zero/empty positions while the "
            "sliding window fills up (warm-up period). Account for this when interpreting "
            "results or sizing positions."
        ),
    )
    n_factors: int = Field(
        ...,
        gt=0,
        description=(
            "Number of latent factors k for the sliding window factor model. "
            "k=1 recovers the single market-factor model; larger k captures finer correlation "
            "structure at the cost of higher estimation noise. "
            "At each streaming step the actual number of components used is "
            "min(n_factors, window, n_valid_assets[, max_components]), so the effective "
            "rank may be lower than n_factors when the number of valid assets or the "
            "window length is the binding constraint."
        ),
    )
    max_components: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional hard cap on the number of SVD components used per streaming step. "
            "When set, the effective component count is "
            "min(n_factors, window, n_valid_assets, max_components). "
            "Useful for large universes where only a few factors dominate and you want to "
            "limit SVD cost below n_factors. Must be >= 1 when provided. Defaults to None."
        ),
    )

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate_max_components(self) -> "SlidingWindowConfig":
        if self.max_components is not None and self.max_components > self.n_factors:
            msg = f"max_components ({self.max_components}) must not exceed n_factors ({self.n_factors})"
            raise ValueError(msg)
        return self


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

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_flat_kwargs(cls, data: dict[str, object]) -> dict[str, object]:
        """Raise an informative TypeError when the pre-v0.4 flat kwargs are used.

        Before v0.4 callers passed ``covariance_mode``, ``n_factors``, and
        ``window`` as top-level keyword arguments to :class:`BasanosConfig`.
        Those fields were replaced by the nested discriminated union
        ``covariance_config``.  Without this validator Pydantic raises a
        generic ``extra_forbidden`` error that gives no migration guidance.

        Examples:
            >>> BasanosConfig(
            ...     vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6,
            ...     covariance_mode="sliding_window", window=30, n_factors=2,
            ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            TypeError: ...
        """
        legacy_keys = {"covariance_mode", "n_factors", "window"}
        found = legacy_keys & data.keys()
        if found:
            found_str = ", ".join(f"'{k}'" for k in sorted(found))
            msg = (
                f"BasanosConfig received legacy keyword argument(s): {found_str}. "
                "These flat fields were removed in v0.4. "
                "Migrate to the nested covariance_config API:\n\n"
                "  # Before (v0.3 and earlier):\n"
                "  BasanosConfig(..., covariance_mode='sliding_window', window=30, n_factors=2)\n\n"
                "  # After (v0.4+):\n"
                "  from basanos.math import SlidingWindowConfig\n"
                "  BasanosConfig(..., covariance_config=SlidingWindowConfig(window=30, n_factors=2))\n\n"
                "For the default EWMA-shrink mode no covariance_config argument is needed."
            )
            raise TypeError(msg)
        return data

    def replace(
        self,
        *,
        vola: int | None = None,
        corr: int | None = None,
        clip: float | None = None,
        shrink: float | None = None,
        aum: float | None = None,
        profit_variance_init: float | None = None,
        profit_variance_decay: float | None = None,
        denom_tol: float | None = None,
        position_scale: float | None = None,
        min_corr_denom: float | None = None,
        max_nan_fraction: float | None = None,
        covariance_config: "CovarianceConfig | None" = None,
    ) -> "BasanosConfig":
        """Return a new :class:`BasanosConfig` with selected fields replaced.

        Unlike :meth:`model_copy`, this method uses explicit constructor kwarg
        forwarding so that any new required field added to
        :class:`BasanosConfig` surfaces immediately as a type or lint error at
        the call site, rather than silently failing at runtime.

        All parameters default to ``None``, meaning *keep the existing value*.
        Pass a non-``None`` value for every field you want to change.

        Args:
            vola: EWMA lookback for volatility normalisation.
            corr: EWMA lookback for correlation estimation.
            clip: Clipping threshold for volatility adjustment.
            shrink: Retention weight λ ∈ [0, 1] for linear shrinkage.
            aum: Assets under management for portfolio scaling.
            profit_variance_init: Initial value for the profit-variance EMA.
            profit_variance_decay: EMA decay factor for realized P&L variance.
            denom_tol: Minimum normalisation denominator.
            position_scale: Multiplicative scaling factor for cash positions.
            min_corr_denom: Guard threshold for the EWMA correlation denominator.
            max_nan_fraction: Maximum tolerated null fraction per price column.
            covariance_config: Covariance estimation configuration.

        Returns:
            A new :class:`BasanosConfig` with the specified fields replaced and
            all other fields copied from ``self``.

        Examples:
            >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
            >>> cfg2 = cfg.replace(shrink=0.8)
            >>> cfg2.shrink
            0.8
            >>> cfg2.vola == cfg.vola
            True
        """
        return BasanosConfig(
            vola=self.vola if vola is None else vola,
            corr=self.corr if corr is None else corr,
            clip=self.clip if clip is None else clip,
            shrink=self.shrink if shrink is None else shrink,
            aum=self.aum if aum is None else aum,
            profit_variance_init=self.profit_variance_init if profit_variance_init is None else profit_variance_init,
            profit_variance_decay=self.profit_variance_decay
            if profit_variance_decay is None
            else profit_variance_decay,
            denom_tol=self.denom_tol if denom_tol is None else denom_tol,
            position_scale=self.position_scale if position_scale is None else position_scale,
            min_corr_denom=self.min_corr_denom if min_corr_denom is None else min_corr_denom,
            max_nan_fraction=self.max_nan_fraction if max_nan_fraction is None else max_nan_fraction,
            covariance_config=self.covariance_config if covariance_config is None else covariance_config,
        )

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
            >>> from basanos.math import BasanosConfig
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
