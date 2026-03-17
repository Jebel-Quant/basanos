"""Async wrappers for :class:`~basanos.math.optimizer.BasanosEngine`.

This module exposes :class:`AsyncBasanosEngine`, a thin, non-blocking facade
around the synchronous :class:`~basanos.math.optimizer.BasanosEngine`.  All
computationally expensive properties are wrapped with :func:`asyncio.to_thread`
so they run in a worker thread and never block the calling event loop.

This is the recommended entry-point for interactive dashboards (e.g. Marimo)
and real-time applications built on top of ``asyncio``.

Examples:
    >>> import asyncio
    >>> import numpy as np
    >>> import polars as pl
    >>> from basanos.math import AsyncBasanosEngine, BasanosConfig
    >>> dates = pl.Series("date", list(range(100)))
    >>> rng0 = np.random.default_rng(0).lognormal(size=100)
    >>> rng1 = np.random.default_rng(1).lognormal(size=100)
    >>> prices = pl.DataFrame({"date": dates, "A": rng0, "B": rng1})
    >>> rng2 = np.random.default_rng(2).normal(size=100)
    >>> rng3 = np.random.default_rng(3).normal(size=100)
    >>> mu = pl.DataFrame({"date": dates, "A": rng2, "B": rng3})
    >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
    >>> engine = AsyncBasanosEngine(prices=prices, mu=mu, cfg=cfg)
    >>> isinstance(engine.assets, list)
    True
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from .optimizer import BasanosConfig, BasanosEngine

if TYPE_CHECKING:
    from ..analytics import Portfolio


class AsyncBasanosEngine:
    """Non-blocking facade around :class:`~basanos.math.optimizer.BasanosEngine`.

    Wraps every expensive property of :class:`~basanos.math.optimizer.BasanosEngine`
    in :func:`asyncio.to_thread` so that CPU-bound correlation and position
    computations are offloaded to a worker thread and the event loop remains
    responsive.

    Construction mirrors :class:`~basanos.math.optimizer.BasanosEngine` exactly:
    the same arguments are accepted and the same validation is performed eagerly
    during ``__init__``.

    Attributes:
        prices: Polars DataFrame of asset prices (must include a ``'date'`` column).
        mu: Polars DataFrame of expected returns with the same shape/columns as
            ``prices`` (must include a ``'date'`` column).
        cfg: :class:`~basanos.math.optimizer.BasanosConfig` instance holding all
            optimizer hyper-parameters.

    Examples:
        >>> import asyncio
        >>> import numpy as np
        >>> import polars as pl
        >>> from basanos.math import AsyncBasanosEngine, BasanosConfig
        >>> dates = pl.Series("date", list(range(100)))
        >>> rng0 = np.random.default_rng(0).lognormal(size=100)
        >>> rng1 = np.random.default_rng(1).lognormal(size=100)
        >>> prices = pl.DataFrame({"date": dates, "A": rng0, "B": rng1})
        >>> rng2 = np.random.default_rng(2).normal(size=100)
        >>> rng3 = np.random.default_rng(3).normal(size=100)
        >>> mu = pl.DataFrame({"date": dates, "A": rng2, "B": rng3})
        >>> cfg = BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1e6)
        >>> engine = AsyncBasanosEngine(prices=prices, mu=mu, cfg=cfg)
        >>> asyncio.run(engine.vola())  # doctest: +ELLIPSIS
        shape: (100, 3)
        ...
    """

    def __init__(self, prices: pl.DataFrame, mu: pl.DataFrame, cfg: BasanosConfig) -> None:
        """Initialise by building the underlying :class:`~basanos.math.optimizer.BasanosEngine`.

        Validation is performed eagerly so that any configuration or data errors
        are surfaced immediately at construction time rather than deferred until
        the first ``await``.

        Args:
            prices: Polars DataFrame of price levels with a ``'date'`` column.
            mu: Polars DataFrame of expected-return signals aligned with ``prices``.
            cfg: :class:`~basanos.math.optimizer.BasanosConfig` with optimizer settings.

        Raises:
            MissingDateColumnError: If ``prices`` or ``mu`` lacks a ``'date'`` column.
            ShapeMismatchError: If ``prices`` and ``mu`` have different shapes.
            ColumnMismatchError: If ``prices`` and ``mu`` have different column names.
            NonPositivePricesError: If any asset column contains non-positive prices.
            ExcessiveNullsError: If any asset column exceeds the maximum null fraction.
            MonotonicPricesError: If any asset column is strictly monotonic.
        """
        self._engine: BasanosEngine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)

    # ── Synchronous passthrough properties ─────────────────────────────────────

    @property
    def prices(self) -> pl.DataFrame:
        """Price DataFrame passed at construction (same as the underlying engine)."""
        return self._engine.prices

    @property
    def mu(self) -> pl.DataFrame:
        """Expected-return DataFrame passed at construction."""
        return self._engine.mu

    @property
    def cfg(self) -> BasanosConfig:
        """Configuration object passed at construction."""
        return self._engine.cfg

    @property
    def assets(self) -> list[str]:
        """List of numeric asset column names (excludes ``'date'``)."""
        return self._engine.assets

    # ── Async methods ───────────────────────────────────────────────────────────

    async def ret_adj(self) -> pl.DataFrame:
        """Return volatility-adjusted log returns without blocking the event loop.

        Delegates to :attr:`~basanos.math.optimizer.BasanosEngine.ret_adj` in a
        worker thread via :func:`asyncio.to_thread`.

        Returns:
            Polars DataFrame of vol-adjusted, clipped log returns aligned with
            ``self.prices``.
        """
        return await asyncio.to_thread(lambda: self._engine.ret_adj)

    async def vola(self) -> pl.DataFrame:
        """Return per-asset EWMA volatility without blocking the event loop.

        Delegates to :attr:`~basanos.math.optimizer.BasanosEngine.vola` in a
        worker thread via :func:`asyncio.to_thread`.

        Returns:
            Polars DataFrame of EWMA volatility estimates aligned with ``self.prices``.
        """
        return await asyncio.to_thread(lambda: self._engine.vola)

    async def cor(self) -> dict[object, np.ndarray]:
        """Return per-timestamp EWM correlation matrices without blocking the event loop.

        Delegates to :attr:`~basanos.math.optimizer.BasanosEngine.cor` in a
        worker thread via :func:`asyncio.to_thread`.  This is typically the most
        expensive single operation (O(T·N²)), so offloading it is especially
        beneficial in interactive applications.

        Returns:
            dict mapping each timestamp to an ``(N, N)`` NumPy correlation matrix.
        """
        return await asyncio.to_thread(lambda: self._engine.cor)

    async def cor_tensor(self) -> np.ndarray:
        """Return all correlation matrices as a 3-D tensor without blocking the event loop.

        Delegates to :attr:`~basanos.math.optimizer.BasanosEngine.cor_tensor` in a
        worker thread via :func:`asyncio.to_thread`.

        Returns:
            NumPy array of shape ``(T, N, N)`` containing the per-timestamp
            correlation matrices.
        """
        return await asyncio.to_thread(lambda: self._engine.cor_tensor)

    async def cash_position(self) -> pl.DataFrame:
        """Return optimized cash positions without blocking the event loop.

        Delegates to :attr:`~basanos.math.optimizer.BasanosEngine.cash_position`
        in a worker thread via :func:`asyncio.to_thread`.

        Returns:
            Polars DataFrame with columns ``['date'] + assets`` containing the
            per-timestamp cash positions.
        """
        return await asyncio.to_thread(lambda: self._engine.cash_position)

    async def portfolio(self) -> Portfolio:
        """Return a fully constructed :class:`~basanos.analytics.Portfolio` without blocking the event loop.

        Delegates to :attr:`~basanos.math.optimizer.BasanosEngine.portfolio` in a
        worker thread via :func:`asyncio.to_thread`.

        Returns:
            :class:`~basanos.analytics.Portfolio` built from the optimized cash positions
            scaled by ``cfg.position_scale`` and ``cfg.aum``.
        """
        return await asyncio.to_thread(lambda: self._engine.portfolio)
