"""Covariance-mode configuration for the Basanos optimizer.

Split out of ``_config.py`` so that module holds `BasanosConfig` alone.
`CovarianceMode`, the two per-mode configs and the `CovarianceConfig`
discriminated union live here; ``_config.py`` re-exports all four, so existing
imports are unaffected.
"""

import enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class CovarianceMode(enum.StrEnum):
    r"""Covariance estimation mode for the Basanos optimizer.

    Attributes:
        ewma_shrink: EWMA correlation matrix with linear shrinkage toward the
            identity.  Controlled by `shrink`.
            This is the default mode.
        sliding_window: Rolling-window factor model.  A fixed block of the
            ``W`` most recent volatility-adjusted returns is decomposed via
            truncated SVD into ``k`` latent factors, giving the estimator

            $$
            \\hat{C}_t^{(W,k)} = \\frac{1}{W}
                \\mathbf{V}_{k,t}\\mathbf{\\Sigma}_{k,t}^2\\mathbf{V}_{k,t}^\\top
                + \\hat{D}_t
            $$

            where $\\hat{D}_t$ is chosen to enforce unit diagonal.
            The system is solved efficiently via the Woodbury identity
            (Section 4.3 of basanos.pdf) at $O(k^3 + kn)$ per step
            rather than $O(n^3)$.
            Configured via `SlidingWindowConfig`.

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
    beyond those already present on `BasanosConfig` (``shrink``, ``corr``).

    .. note::
        This class is **intentionally minimal**. The only field is the
        ``covariance_mode`` discriminator, which is required to make Pydantic's
        discriminated-union dispatch work correctly (see `CovarianceConfig`).
        Before adding new EWMA-specific fields here, consider whether the field
        name clashes with existing `BasanosConfig` top-level fields and
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

    $$
    k_{\text{eff}} = \min(k,\; W,\; n_{\text{valid}},\; k_{\text{max}})
    $$

    where $k$ = ``n_factors``, $W$ = ``window``,
    $n_{\text{valid}}$ is the number of assets with finite prices at that
    step, and $k_{\text{max}}$ = ``max_components`` (or $+\infty$
    when unset).  This ensures the truncated SVD remains well-posed even when
    assets temporarily drop out of the universe.  Setting ``max_components``
    explicitly caps computational cost in large universes without changing the
    desired factor count used in batch mode.

    Args:
        window: Rolling window length $W \\geq 1$.
            Rule of thumb: $W \\geq 2n$ keeps the sample covariance
            well-posed before truncation.
        n_factors: Number of latent factors $k \\geq 1$.
            $k = 1$ recovers the single market-factor model; larger
            $k$ captures finer correlation structure at the cost of
            higher estimation noise.
        max_components: Optional hard cap on the number of SVD components used
            per streaming step.  When set, the effective component count is
            $\\min(k_{\\text{eff}},\\, \\texttt{max\\_components})$.
            Useful for large universes where only a few factors dominate and
            you want to limit SVD cost below ``n_factors``.  Must be
            $\\geq 1$ when provided.  Defaults to ``None`` (no extra cap).

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
        """Validate that max_components does not exceed n_factors."""
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

* `EwmaShrinkConfig` when ``covariance_mode="ewma_shrink"``
* `SlidingWindowConfig` when ``covariance_mode="sliding_window"``
"""
