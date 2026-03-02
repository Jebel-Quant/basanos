"""Tests for Portfolio.smoothed_holding with n=0.

This test ensures the early-return branch (n == 0) is exercised so that
coverage reaches 100% in analytics/portfolio.py. The method should return
`self` (same instance), and no data should be changed.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import polars.testing as pt

from basanos.analytics import Portfolio


def _make_prices_positions(n: int = 10) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Construct small deterministic price and position frames with a date column."""
    start = date(2020, 1, 1)
    end = start + timedelta(days=n - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    prices = pl.DataFrame(
        {
            "date": dates,
            # simple linear trends to produce non-zero pct_change
            "A": pl.Series([100.0 + 1.0 * i for i in range(n)], dtype=pl.Float64),
            "B": pl.Series([200.0 - 0.5 * i for i in range(n)], dtype=pl.Float64),
        }
    )
    positions = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([1000.0 + 2.0 * i for i in range(n)], dtype=pl.Float64),
            "B": pl.Series([500.0 + (i % 2) for i in range(n)], dtype=pl.Float64),
        }
    )
    return prices, positions


def test_smoothed_holding_zero_returns_self_and_preserves_state():
    """Calling smoothed_holding(0) should return the same Portfolio instance and keep data intact."""
    prices, positions = _make_prices_positions(n=12)
    pf = Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e6)

    pf_zero = pf.smoothed_holding(0)

    # Identity: method returns self when n == 0
    assert pf_zero is pf

    # Data unchanged
    assert pf_zero.aum == pf.aum
    pt.assert_frame_equal(pf_zero.prices, prices)
    pt.assert_frame_equal(pf_zero.cashposition, positions)

    # Downstream computation still works (and doesn't mutate state)
    nav_before = pf.nav_accumulated
    _ = pf.profit
    nav_after = pf_zero.nav_accumulated
    pt.assert_frame_equal(nav_after, nav_before)
