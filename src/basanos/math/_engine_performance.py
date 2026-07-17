"""Performance / parameter-sweep mixin for `BasanosEngine`.

Provides the Sharpe-ratio sweep helpers (`sharpe_at_shrink`,
`sharpe_at_window_factors`, `naive_sharpe`) as a reusable mixin so that
``optimizer.py`` stays focused on the position-solving facade.  Each helper
rebuilds a sibling engine with a modified configuration, so it constructs a new
`BasanosEngine` via a deferred import (avoiding a circular import at module
load time).

Classes in this module are **private implementation details**.  The public API
is `BasanosEngine`, which inherits from `_PerformanceMixin`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ._config import SlidingWindowConfig

if TYPE_CHECKING:
    from ._engine_protocol import _EngineProtocol


class _PerformanceMixin:
    """Mixin providing portfolio-Sharpe sweep helpers for `BasanosEngine`.

    The consuming class must satisfy `_EngineProtocol` (it uses ``assets``,
    ``prices``, ``mu``, and ``cfg``).
    """

    def sharpe_at_shrink(self: _EngineProtocol, shrink: float) -> float:
        r"""Return the annualised portfolio Sharpe ratio for the given shrinkage weight.

        Constructs a new `BasanosEngine` with all parameters identical to
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
                `shrink` for full documentation.

        Returns:
            Annualised Sharpe ratio of the portfolio returns as a ``float``.
            Returns ``float("nan")`` when the Sharpe ratio cannot be computed
            (e.g. zero-variance returns).

        Raises:
            ValidationError: When ``shrink`` is outside [0, 1] (delegated to
                `BasanosConfig` field validation).

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
        from .optimizer import BasanosEngine  # deferred to avoid a circular import

        new_cfg = self.cfg.replace(shrink=shrink)
        engine = BasanosEngine(prices=self.prices, mu=self.mu, cfg=new_cfg)
        return float(engine.portfolio.stats.sharpe().get("returns") or float("nan"))

    def sharpe_at_window_factors(self: _EngineProtocol, window: int, n_factors: int) -> float:
        r"""Return the annualised portfolio Sharpe ratio for the given sliding-window parameters.

        Constructs a new `BasanosEngine` with ``covariance_mode`` set to
        ``"sliding_window"`` and the supplied ``window`` / ``n_factors``, keeping
        all other configuration identical to ``self``.

        Use this method to sweep ``(W, k)`` and compare the sliding-window
        estimator against the EWMA baseline (via `sharpe_at_shrink`).

        Args:
            window: Rolling window length $W \geq 1$.
                Rule of thumb: $W \geq 2 \cdot n_{\text{assets}}$.
            n_factors: Number of latent factors $k \geq 1$.

        Returns:
            Annualised Sharpe ratio of the portfolio returns as a ``float``.
            Returns ``float("nan")`` when the Sharpe ratio cannot be computed
            (e.g. not enough history to fill the first window).

        Raises:
            ValidationError: When ``window`` or ``n_factors`` fail field
                constraints (delegated to `BasanosConfig`).

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
        from .optimizer import BasanosEngine  # deferred to avoid a circular import

        new_cfg = self.cfg.replace(
            covariance_config=SlidingWindowConfig(window=window, n_factors=n_factors),
        )
        engine = BasanosEngine(prices=self.prices, mu=self.mu, cfg=new_cfg)
        return float(engine.portfolio.stats.sharpe().get("returns") or float("nan"))

    @property
    def naive_sharpe(self: _EngineProtocol) -> float:
        r"""Sharpe ratio of the naïve equal-weight signal (μ = 1 for every asset/timestamp).

        Replaces the expected-return signal ``mu`` with a constant matrix of
        ones, then runs the optimiser with the current configuration and returns
        the annualised Sharpe ratio of the resulting portfolio.

        This provides the baseline answer to *"does the signal add value?"*:
        a real signal should produce a higher Sharpe than the naïve benchmark.
        Combined with `sharpe_at_shrink`, this yields a three-way
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
        from .optimizer import BasanosEngine  # deferred to avoid a circular import

        naive_mu = self.mu.with_columns(pl.lit(1.0).alias(asset) for asset in self.assets)
        engine = BasanosEngine(prices=self.prices, mu=naive_mu, cfg=self.cfg)
        return float(engine.portfolio.stats.sharpe().get("returns") or float("nan"))
