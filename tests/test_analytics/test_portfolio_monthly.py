"""Tests for Portfolio.monthly compounded returns.

These tests validate that the monthly property returns end-of-month dates
and compounded monthly returns derived from daily profit/AUM, matching
analytical expectations on a deterministic dataset.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from basanos.analytics import Portfolio


@pytest.fixture
def portfolio():
    """Build a small deterministic Portfolio for monthly aggregation tests."""
    start = date(2020, 1, 10)
    days = 80
    end = start + timedelta(days=days - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True).cast(pl.Date)

    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, size=days)
    prices = (1.0 + returns).cumprod()

    # make a frame of prices
    prices = pl.DataFrame({"date": dates, "A": prices})

    cash_position = pl.DataFrame({"date": dates, "A": pl.Series([1000.0] * days, dtype=pl.Float64)})

    portfolio = Portfolio(prices=prices, cashposition=cash_position, aum=10000)
    return portfolio


def test_monthly_structure_and_end_of_month_dates(portfolio):
    """Monthly should include date (month-end), returns, and calendar columns including month_name."""
    monthly = portfolio.monthly

    # Columns and types
    assert monthly.columns == ["date", "returns", "NAV_accumulated", "profit", "year", "month", "month_name"]
    assert monthly["date"].dtype == pl.Date

    expected_years = [2020, 2020, 2020]
    assert list(monthly["year"]) == expected_years

    expected_months = [1, 2, 3]
    assert list(monthly["month"]) == expected_months

    expected_month_names = ["Jan", "Feb", "Mar"]
    assert list(monthly["month_name"]) == expected_month_names

    # Returns should be finite
    assert monthly["returns"].is_finite().all()
