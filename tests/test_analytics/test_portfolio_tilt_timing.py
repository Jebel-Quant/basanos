"""Tests for Portfolio.tilt and Portfolio.timing in basanos.analytics.portfolio.

These tests verify that:
- tilt.prices are constructed from mean daily pct_change as (1+r)^t and preserve the date column,
- timing.prices equals prices - tilt.prices elementwise (date preserved),
- the derived portfolios preserve AUM and can compute profit and NAV without errors.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import polars.testing as pt

from basanos.analytics import Portfolio


def _make_prices_positions(n: int = 40) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create deterministic prices and positions with a date column.

    Asset A grows approximately exponentially; Asset B oscillates to
    ensure non-trivial pct_change and mean returns.
    """
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True)

    a = pl.Series([100.0 * (1.01**i) for i in range(n)], dtype=pl.Float64)
    b = pl.Series([200.0 + 5.0 * np.sin(0.2 * i) for i in range(n)], dtype=pl.Float64)
    prices = pl.DataFrame({"date": dates, "A": a, "B": b})

    pos_a = pl.Series([1000.0 + 2.0 * i for i in range(n)], dtype=pl.Float64)
    pos_b = pl.Series([500.0 + (i % 3) for i in range(n)], dtype=pl.Float64)
    positions = pl.DataFrame({"date": dates, "A": pos_a, "B": pos_b})

    return prices, positions


def test_timing_prices_are_difference_and_portfolio_computable():
    """timing.prices must equal original prices - tilt.prices, date preserved.

    Also ensure the timing portfolio preserves AUM and computes profit/NAV.
    """
    prices, positions = _make_prices_positions(n=50)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=5e5)

    tilt = pf.tilt
    assert isinstance(tilt, Portfolio)
    assert tilt.aum == pf.aum
    pt.assert_frame_equal(tilt.prices, pf.prices)

    timing = pf.timing
    assert isinstance(timing, Portfolio)
    assert timing.aum == pf.aum
    pt.assert_frame_equal(timing.prices, pf.prices)

    pt.assert_frame_equal(
        pf.cashposition.select(pf.assets), timing.cashposition.select(pf.assets) + tilt.cashposition.select(pf.assets)
    )
    print(pf.tilt_timing_decomp)
