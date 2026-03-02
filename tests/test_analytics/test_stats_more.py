"""Additional tests for basanos.analytics.stats.Stats to reach 100% coverage.

These tests target branches/lines not exercised by the primary test suite:
- assets property: filters out the 'date' column and non-numeric columns.
- periods_per_year: computes periods/year from daily timestamps.
- volatility default path: uses periods_per_year when periods is None.
- _to_float: None and timedelta inputs.
- _to_float_or_none: None and timedelta inputs.
- periods_per_year: numeric first column and single-row fallback paths.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl

from basanos.analytics._stats import Stats, _to_float, _to_float_or_none


def _make_daily_frame(days: int = 10) -> pl.DataFrame:
    """Build a simple DataFrame with a daily 'date' column and one numeric asset.

    The numeric series is arbitrary; only the index cadence matters for
    periods_per_year and volatility scaling checks.
    """
    start = date(2020, 1, 1)
    end = start + timedelta(days=days - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True)
    # Simple deterministic series
    a = pl.Series("A", [float(i) for i in range(days)], dtype=pl.Float64)
    return pl.DataFrame({"date": dates, "A": a, "label": ["x"] * days})


def test_assets_property_filters_date_and_non_numeric():
    """Assets should include only numeric columns and exclude 'date' and strings."""
    df = _make_daily_frame(5)
    s = Stats(df)
    assert s.assets == ["A"]


def test_periods_per_year_estimation_for_daily_index():
    """periods_per_year should be approximately 365 for daily timestamps."""
    df = _make_daily_frame(8)
    s = Stats(df)
    ppy = s.periods_per_year
    assert math.isfinite(ppy)
    assert abs(ppy - 365.0) < 1e-6


def test_volatility_uses_default_periods_per_year_when_none():
    """volatility() without periods should scale by sqrt(periods_per_year)."""
    df = _make_daily_frame(12)
    s = Stats(df)
    base_std = float(df["A"].std())
    expected = base_std * math.sqrt(s.periods_per_year)
    out = s.volatility()  # periods=None, annualize=True (default)
    assert abs(out["A"] - expected) < 1e-12


# --- Tests for private helper functions ---


def test_to_float_none_returns_zero():
    """_to_float(None) should return 0.0."""
    assert _to_float(None) == 0.0


def test_to_float_timedelta_returns_total_seconds():
    """_to_float(timedelta) should return total_seconds() as float."""
    assert _to_float(timedelta(seconds=42)) == 42.0


def test_to_float_or_none_none_returns_none():
    """_to_float_or_none(None) should return None."""
    assert _to_float_or_none(None) is None


def test_to_float_or_none_timedelta_returns_total_seconds():
    """_to_float_or_none(timedelta) should return total_seconds() as float."""
    assert _to_float_or_none(timedelta(seconds=7)) == 7.0


def test_periods_per_year_single_row_uses_fallback():
    """Single-row frame: diff is empty → mean is None → fallback to 86400 s → 365 days/year."""
    df = pl.DataFrame(
        {
            "date": pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 1), interval="1d", eager=True),
            "A": [1.0],
        }
    )
    s = Stats(df)
    # fallback: seconds = 86400 → ppy = 365*24*3600 / 86400 = 365.0
    assert math.isclose(s.periods_per_year, 365.0, rel_tol=1e-12)


def test_periods_per_year_numeric_first_column_uses_to_float():
    """Numeric first column: mean_diff is a float, not timedelta → elif branch executed."""
    df = pl.DataFrame({"steps": [1, 2, 3], "A": [1.0, 2.0, 3.0]})
    s = Stats(df)
    # diff values: [1, 1], mean = 1.0 second-equivalent
    # ppy = 365*24*3600 / 1.0
    expected = 365.0 * 24.0 * 3600.0
    assert math.isclose(s.periods_per_year, expected, rel_tol=1e-12)
