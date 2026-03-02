"""Tests for Portfolio.truncate in basanos.analytics.portfolio.

These tests verify that truncation by date bounds returns a new Portfolio with
filtered data, preserves AUM, and remains usable for downstream computations.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from basanos.analytics import Portfolio


def _make_prices_positions(n: int = 5) -> tuple[pl.DataFrame, pl.DataFrame]:
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([100.0 + 10.0 * i for i in range(n)], dtype=pl.Float64),
            "B": pl.Series([200.0 - 5.0 * i for i in range(n)], dtype=pl.Float64),
        }
    )
    # Hold constant positions to make profit math simple
    positions = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([1000.0] * n, dtype=pl.Float64),
            "B": pl.Series([500.0] * n, dtype=pl.Float64),
        }
    )
    return prices, positions


def test_truncate_by_start_end_inclusive_preserves_aum_and_dates():
    """Truncating with both start and end returns new Portfolio, preserves AUM, and filters dates inclusively."""
    prices, positions = _make_prices_positions(n=5)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    start = date(2020, 1, 2)
    end = date(2020, 1, 4)

    pf_t = pf.truncate(start=start, end=end)

    # A new instance returned and AUM preserved
    assert isinstance(pf_t, Portfolio)
    assert pf_t.aum == pf.aum

    # Height and dates restricted to [start, end]
    assert pf_t.prices.height == 3
    assert pf_t.cashposition.height == 3
    assert pf_t.prices["date"].min() == start
    assert pf_t.prices["date"].max() == end

    # Downstream computation still works
    nav = pf_t.nav_accumulated
    assert "NAV_accumulated" in nav.columns
    assert nav.height == 3


def test_truncate_with_only_start_or_end_open_bounds():
    """Truncating with only a start or only an end applies open bounds and remains computable."""
    prices, positions = _make_prices_positions(n=6)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5)

    # Only start
    start = date(2020, 1, 4)
    pf_s = pf.truncate(start=start)
    assert pf_s.prices["date"].min() == start
    assert pf_s.prices.height == 3  # days 4,5,6
    # Only end
    end = date(2020, 1, 3)
    pf_e = pf.truncate(end=end)
    assert pf_e.prices["date"].max() == end
    assert pf_e.prices.height == 3  # days 1,2,3

    # Profits compute without error on truncated portfolios
    _ = pf_s.profit
    _ = pf_e.profit
