"""Async wrappers for :class:`~basanos.analytics.portfolio.Portfolio`.

This module exposes :class:`AsyncPortfolio`, a thin, non-blocking facade around
the synchronous :class:`~basanos.analytics.portfolio.Portfolio`.  All
computationally intensive properties are wrapped with :func:`asyncio.to_thread`
so they run in a worker thread and never block the calling event loop.

This is the recommended entry-point for interactive dashboards (e.g. Marimo)
and real-time applications built on top of ``asyncio``.

Examples:
    >>> import asyncio
    >>> import polars as pl
    >>> from datetime import date
    >>> from basanos.analytics import AsyncPortfolio, Portfolio
    >>> prices = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [100.0, 110.0]})
    >>> pos = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [1000.0, 1000.0]})
    >>> pf = AsyncPortfolio(Portfolio(prices=prices, cashposition=pos))
    >>> isinstance(pf.assets, list)
    True
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import polars as pl

from .portfolio import Portfolio

if TYPE_CHECKING:
    from ._stats import Stats


class AsyncPortfolio:
    """Non-blocking facade around :class:`~basanos.analytics.portfolio.Portfolio`.

    Wraps every expensive property and method of
    :class:`~basanos.analytics.portfolio.Portfolio` in :func:`asyncio.to_thread`
    so that P&L, NAV, and statistical computations are offloaded to a worker
    thread and the event loop remains responsive.

    Construction accepts a pre-built :class:`~basanos.analytics.portfolio.Portfolio`
    instance.  Convenience class-methods mirror those of the underlying class so
    the two can be used interchangeably at the call site.

    Attributes:
        _portfolio: The wrapped synchronous :class:`~basanos.analytics.portfolio.Portfolio`.

    Examples:
        >>> import asyncio
        >>> import polars as pl
        >>> from datetime import date
        >>> from basanos.analytics import AsyncPortfolio, Portfolio
        >>> prices = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [100.0, 110.0]})
        >>> pos = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "A": [1000.0, 1000.0]})
        >>> pf = AsyncPortfolio(Portfolio(prices=prices, cashposition=pos))
        >>> asyncio.run(pf.profit())  # doctest: +ELLIPSIS
        shape: (2, 2)
        ...
    """

    def __init__(self, portfolio: Portfolio) -> None:
        """Wrap an existing :class:`~basanos.analytics.portfolio.Portfolio`.

        Args:
            portfolio: A fully constructed synchronous
                :class:`~basanos.analytics.portfolio.Portfolio` instance.
        """
        self._portfolio: Portfolio = portfolio

    # ── Class-method mirrors ────────────────────────────────────────────────────

    @classmethod
    def from_risk_position(
        cls,
        prices: pl.DataFrame,
        risk_position: pl.DataFrame,
        vola: int = 32,
        aum: float = 1e8,
    ) -> AsyncPortfolio:
        """Create an :class:`AsyncPortfolio` from per-asset risk positions.

        Mirrors :meth:`~basanos.analytics.portfolio.Portfolio.from_risk_position`.

        Args:
            prices: Price levels per asset over time (may include a date column).
            risk_position: Risk units per asset aligned with prices.
            vola: EWMA lookback (span-equivalent) in trading days.
            aum: Assets under management used as the base NAV offset.

        Returns:
            :class:`AsyncPortfolio` wrapping the constructed Portfolio.
        """
        return cls(Portfolio.from_risk_position(prices=prices, risk_position=risk_position, vola=vola, aum=aum))

    @classmethod
    def from_cash_position(
        cls,
        prices: pl.DataFrame,
        cash_position: pl.DataFrame,
        aum: float = 1e8,
    ) -> AsyncPortfolio:
        """Create an :class:`AsyncPortfolio` directly from cash positions.

        Mirrors :meth:`~basanos.analytics.portfolio.Portfolio.from_cash_position`.

        Args:
            prices: Price levels per asset over time (may include a date column).
            cash_position: Cash exposure per asset over time aligned with prices.
            aum: Assets under management used as the base NAV offset.

        Returns:
            :class:`AsyncPortfolio` wrapping the constructed Portfolio.
        """
        return cls(Portfolio.from_cash_position(prices=prices, cash_position=cash_position, aum=aum))

    # ── Synchronous passthrough properties ─────────────────────────────────────

    @property
    def assets(self) -> list[str]:
        """List the numeric asset column names (excludes ``'date'``)."""
        return self._portfolio.assets

    @property
    def prices(self) -> pl.DataFrame:
        """Price DataFrame held by the underlying portfolio."""
        return self._portfolio.prices

    @property
    def cashposition(self) -> pl.DataFrame:
        """Cash-position DataFrame held by the underlying portfolio."""
        return self._portfolio.cashposition

    @property
    def aum(self) -> float:
        """Assets under management held by the underlying portfolio."""
        return self._portfolio.aum

    # ── Async property equivalents ──────────────────────────────────────────────

    async def profits(self) -> pl.DataFrame:
        """Return per-asset daily P&L without blocking the event loop.

        Delegates to :attr:`~basanos.analytics.portfolio.Portfolio.profits`
        in a worker thread via :func:`asyncio.to_thread`.

        Returns:
            Polars DataFrame of per-asset daily cash profits.
        """
        return await asyncio.to_thread(lambda: self._portfolio.profits)

    async def profit(self) -> pl.DataFrame:
        """Return aggregate daily portfolio profit without blocking the event loop.

        Delegates to :attr:`~basanos.analytics.portfolio.Portfolio.profit`
        in a worker thread via :func:`asyncio.to_thread`.

        Returns:
            Polars DataFrame with a ``'profit'`` column (and ``'date'`` when present).
        """
        return await asyncio.to_thread(lambda: self._portfolio.profit)

    async def nav_accumulated(self) -> pl.DataFrame:
        """Return cumulative additive NAV without blocking the event loop.

        Delegates to :attr:`~basanos.analytics.portfolio.Portfolio.nav_accumulated`
        in a worker thread via :func:`asyncio.to_thread`.

        Returns:
            Polars DataFrame with columns ``['date', 'profit', 'NAV_accumulated']``.
        """
        return await asyncio.to_thread(lambda: self._portfolio.nav_accumulated)

    async def returns(self) -> pl.DataFrame:
        """Return daily returns scaled by AUM without blocking the event loop.

        Delegates to :attr:`~basanos.analytics.portfolio.Portfolio.returns`
        in a worker thread via :func:`asyncio.to_thread`.

        Returns:
            Polars DataFrame with ``'profit'``, ``'NAV_accumulated'``, and
            ``'returns'`` columns.
        """
        return await asyncio.to_thread(lambda: self._portfolio.returns)

    async def monthly(self) -> pl.DataFrame:
        """Return monthly compounded returns without blocking the event loop.

        Delegates to :attr:`~basanos.analytics.portfolio.Portfolio.monthly`
        in a worker thread via :func:`asyncio.to_thread`.

        Returns:
            Polars DataFrame with monthly aggregates including ``'returns'``,
            ``'NAV_accumulated'``, ``'profit'``, ``'year'``, ``'month'``, and
            ``'month_name'``.
        """
        return await asyncio.to_thread(lambda: self._portfolio.monthly)

    async def nav_compounded(self) -> pl.DataFrame:
        """Return compounded NAV without blocking the event loop.

        Delegates to :attr:`~basanos.analytics.portfolio.Portfolio.nav_compounded`
        in a worker thread via :func:`asyncio.to_thread`.

        Returns:
            Polars DataFrame with a ``'NAV_compounded'`` column.
        """
        return await asyncio.to_thread(lambda: self._portfolio.nav_compounded)

    async def highwater(self) -> pl.DataFrame:
        """Return the running NAV maximum without blocking the event loop.

        Delegates to :attr:`~basanos.analytics.portfolio.Portfolio.highwater`
        in a worker thread via :func:`asyncio.to_thread`.

        Returns:
            Polars DataFrame with a ``'highwater'`` column.
        """
        return await asyncio.to_thread(lambda: self._portfolio.highwater)

    async def drawdown(self) -> pl.DataFrame:
        """Return drawdown from the high-water mark without blocking the event loop.

        Delegates to :attr:`~basanos.analytics.portfolio.Portfolio.drawdown`
        in a worker thread via :func:`asyncio.to_thread`.

        Returns:
            Polars DataFrame with ``'drawdown'`` and ``'drawdown_pct'`` columns.
        """
        return await asyncio.to_thread(lambda: self._portfolio.drawdown)

    async def all(self) -> pl.DataFrame:
        """Return a merged view of drawdown and compounded NAV without blocking the event loop.

        Delegates to :attr:`~basanos.analytics.portfolio.Portfolio.all`
        in a worker thread via :func:`asyncio.to_thread`.

        Returns:
            Polars DataFrame containing drawdown metrics and ``'NAV_compounded'``.
        """
        return await asyncio.to_thread(lambda: self._portfolio.all)

    async def stats(self) -> Stats:
        """Return a :class:`~basanos.analytics._stats.Stats` instance without blocking the event loop.

        Delegates to :attr:`~basanos.analytics.portfolio.Portfolio.stats`
        in a worker thread via :func:`asyncio.to_thread`.

        Returns:
            :class:`~basanos.analytics._stats.Stats` built from daily portfolio returns.
        """
        return await asyncio.to_thread(lambda: self._portfolio.stats)

    async def tilt(self) -> AsyncPortfolio:
        """Return the tilt portfolio (constant average weights) without blocking the event loop.

        Delegates to :attr:`~basanos.analytics.portfolio.Portfolio.tilt`
        in a worker thread via :func:`asyncio.to_thread`.

        Returns:
            :class:`AsyncPortfolio` with constant average cash positions.
        """
        result = await asyncio.to_thread(lambda: self._portfolio.tilt)
        return AsyncPortfolio(result)

    async def timing(self) -> AsyncPortfolio:
        """Return the timing portfolio (deviation from tilt) without blocking the event loop.

        Delegates to :attr:`~basanos.analytics.portfolio.Portfolio.timing`
        in a worker thread via :func:`asyncio.to_thread`.

        Returns:
            :class:`AsyncPortfolio` capturing dynamic timing effects.
        """
        result = await asyncio.to_thread(lambda: self._portfolio.timing)
        return AsyncPortfolio(result)

    async def tilt_timing_decomp(self) -> pl.DataFrame:
        """Return the tilt/timing NAV decomposition without blocking the event loop.

        Delegates to :attr:`~basanos.analytics.portfolio.Portfolio.tilt_timing_decomp`
        in a worker thread via :func:`asyncio.to_thread`.

        Returns:
            Polars DataFrame with ``'portfolio'``, ``'tilt'``, and ``'timing'``
            NAV columns.
        """
        return await asyncio.to_thread(lambda: self._portfolio.tilt_timing_decomp)

    async def correlation(self, frame: pl.DataFrame, name: str = "portfolio") -> pl.DataFrame:
        """Return a correlation matrix of asset returns plus the portfolio without blocking the event loop.

        Delegates to :meth:`~basanos.analytics.portfolio.Portfolio.correlation`
        in a worker thread via :func:`asyncio.to_thread`.

        Args:
            frame: Polars DataFrame of asset prices (date column is ignored).
            name: Column name for the portfolio profit series. Defaults to ``'portfolio'``.

        Returns:
            Square Polars DataFrame of Pearson correlations.
        """
        return await asyncio.to_thread(lambda: self._portfolio.correlation(frame, name))

    async def truncate(self, start: object = None, end: object = None) -> AsyncPortfolio:
        """Return a date/row-range truncated portfolio without blocking the event loop.

        Delegates to :meth:`~basanos.analytics.portfolio.Portfolio.truncate`
        in a worker thread via :func:`asyncio.to_thread`.

        Args:
            start: Inclusive lower bound (date/datetime or row index).
            end: Inclusive upper bound (date/datetime or row index).

        Returns:
            :class:`AsyncPortfolio` filtered to the requested range.
        """
        result = await asyncio.to_thread(lambda: self._portfolio.truncate(start, end))
        return AsyncPortfolio(result)

    async def lag(self, n: int) -> AsyncPortfolio:
        """Return a portfolio with cash positions lagged by ``n`` steps without blocking.

        Delegates to :meth:`~basanos.analytics.portfolio.Portfolio.lag`
        in a worker thread via :func:`asyncio.to_thread`.

        Args:
            n: Number of rows to shift (can be negative, zero, or positive).

        Returns:
            :class:`AsyncPortfolio` with lagged cash positions.
        """
        result = await asyncio.to_thread(lambda: self._portfolio.lag(n))
        return AsyncPortfolio(result)

    async def smoothed_holding(self, n: int) -> AsyncPortfolio:
        """Return a portfolio with cash positions smoothed by a rolling mean without blocking.

        Delegates to :meth:`~basanos.analytics.portfolio.Portfolio.smoothed_holding`
        in a worker thread via :func:`asyncio.to_thread`.

        Args:
            n: Number of previous steps to include in the rolling average.

        Returns:
            :class:`AsyncPortfolio` with smoothed cash positions.
        """
        result = await asyncio.to_thread(lambda: self._portfolio.smoothed_holding(n))
        return AsyncPortfolio(result)
