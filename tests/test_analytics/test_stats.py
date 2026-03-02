"""Tests for Stats in taipan.quantstats._stats.

This module provides comprehensive tests for all methods of the Stats class,
using small deterministic datasets and explicit expectations. Docstrings are
included to satisfy pydocstyle.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from basanos.analytics._stats import Stats


@pytest.fixture
def dates() -> list[datetime]:
    """Return a list of consecutive daily datetimes used as a test index."""
    start = "2020-01-01"
    days = 200
    dt0 = datetime.fromisoformat(start)
    dates = [dt0 + timedelta(days=i) for i in range(days)]
    return dates


@pytest.fixture
def frame(dates: list[datetime]) -> pl.DataFrame:
    """Build a Polars DataFrame with a 'date' column from the dates fixture."""
    return pl.DataFrame({"date": dates})


def test_skew_symmetric_zero(frame):
    """Symmetric distribution around zero should have skew approximately zero."""
    data = frame.head(8)
    a = [-3.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 3.0]
    data = data.with_columns(pl.Series("A", a))
    s = Stats(data)

    result = s.skew()

    assert set(result.keys()) == {"A"}
    assert math.isclose(result["A"], 0.0, abs_tol=1e-12, rel_tol=1e-9)


def test_skew_positive_and_negative_signs(frame):
    """Positively skewed data yields positive skew; negatively skewed yields negative skew."""
    data = frame.head(6)

    a = [-0.5, -0.2, -0.1, 0.0, 0.0, 3.0]
    b = [0.5, 0.2, 0.1, 0.0, 0.0, -3.0]

    data = data.with_columns(pl.Series("A", a), pl.Series("B", b))

    s = Stats(data)
    out = s.skew()

    assert out["A"] > 0
    assert out["B"] < 0
    assert math.isclose(abs(out["A"]), abs(out["B"]), rel_tol=0.25)


def test_skew_ignores_nulls_and_returns_dict(frame):
    """Skew should ignore nulls and return a mapping of asset->float values."""
    data = frame.head(7)
    a = [None, 0.0, 0.1, None, -0.1, 0.2, -0.2]

    data = data.with_columns(pl.Series("A", a))
    s = Stats(data)

    res: dict[str, Any] = s.skew()

    assert isinstance(res, dict)
    assert set(res) == {"A"}
    assert isinstance(res["A"], float)
    assert math.isfinite(res["A"])
    assert res["A"] == pytest.approx(0.0, abs=1e-12)


def test_kurtosis_heavy_and_flat_signs(frame):
    """Kurtosis should be positive for heavy tails and negative for flat/uniform-like data."""
    data = frame.head(10)

    a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, -10.0, 0.0]
    b = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    data = data.with_columns(pl.Series("A", a), pl.Series("B", b))
    s = Stats(data)

    out = s.kurtosis()

    assert out["A"] > 0
    assert out["B"] < 0


def test_kurtosis_ignores_nulls_and_returns_dict(frame):
    """Kurtosis should ignore nulls and return a mapping of asset->float values."""
    data = frame.head(8)
    a = [None, -1.0, 0.0, 1.0, None, -2.0, 2.0, 0.0]

    data = data.with_columns(pl.Series("A", a))
    s = Stats(data)

    res = s.kurtosis()

    assert isinstance(res, dict)
    assert set(res.keys()) == {"A"}
    assert isinstance(res["A"], float)
    assert math.isfinite(res["A"])


def test_avg_return_win_loss(frame):
    """avg_return ignores zeros/nulls; avg_win/loss compute means of positive/negative subsets."""
    data = frame.head(6)
    a = [0.1, 0.0, -0.2, None, 0.3, -0.4]
    rets = pl.DataFrame({"A": a})
    data = data.with_columns(pl.Series("A", a))
    s = Stats(data)

    out_avg = s.avg_return()
    out_win = s.avg_win()
    out_loss = s.avg_loss()

    expected_avg = rets["A"].filter(rets["A"].is_not_null() & (rets["A"] != 0)).mean()
    expected_win = rets["A"].filter(rets["A"] > 0).mean()
    expected_loss = rets["A"].filter(rets["A"] < 0).mean()

    assert out_avg["A"] == expected_avg
    assert out_win["A"] == expected_win
    assert out_loss["A"] == expected_loss


# --- Volatility and Sharpe ---


def test_volatility_paths(frame):
    """Volatility should annualize by sqrt(periods) when requested and accept custom periods."""
    data = frame.head(5)
    a = [0.01, 0.02, 0.00, -0.01, 0.03]
    rets = pl.DataFrame({"A": a})
    data = data.with_columns(pl.Series("A", a))
    s = Stats(data)

    base_std = float(rets["A"].std())
    out_no = s.volatility(annualize=False)
    assert math.isclose(out_no["A"], base_std, rel_tol=1e-12, abs_tol=1e-12)

    out_252 = s.volatility(periods=252, annualize=True)
    assert math.isclose(out_252["A"], base_std * math.sqrt(252), rel_tol=1e-12, abs_tol=1e-12)

    out_60 = s.volatility(periods=60, annualize=True)
    assert math.isclose(out_60["A"], base_std * math.sqrt(60), rel_tol=1e-12, abs_tol=1e-12)


def test_sharpe_simple(frame):
    """Sharpe equals mean/std (ddof=1) scaled by sqrt(periods)."""
    data = frame.head(5)
    a = [0.01, 0.02, 0.00, -0.01, 0.03]
    rets = pl.DataFrame({"A": a})
    data = data.with_columns(pl.Series("A", a))
    s = Stats(data)

    mean = float(rets["A"].mean())
    std_sample = float(rets["A"].std(ddof=1))
    expected = mean / std_sample  # with periods=1

    out = s.sharpe(periods=1)
    assert math.isclose(out["A"], expected, rel_tol=1e-12, abs_tol=1e-12)


# --- Risk (VaR / CVaR) ---


def test_var_and_cvar_properties(frame):
    """VaR(alpha) should be more negative than mean; CVaR should be <= VaR for normal-like data."""
    data = frame.head(200)
    # Create roughly normal-like returns around 0 with small variance
    # Use a deterministic sequence scaled down
    base = np.linspace(-3, 3, num=200)
    a = (base / 100.0).tolist()
    pl.DataFrame({"A": a})
    data = data.with_columns(pl.Series("A", a))
    s = Stats(data)

    var = s.value_at_risk(alpha=0.05)["A"]
    cvar = s.conditional_value_at_risk(alpha=0.05)["A"]

    # VaR should be near left-tail quantile (negative)
    assert var < 0
    # CVaR is the average of the tail <= VaR, thus even more negative or equal
    assert cvar <= var


# --- Win rate, best/worst, exposure ---


def test_volatility_invalid_periods_raises_typeerror(frame):
    """Passing a non-numeric periods value to volatility should raise TypeError."""
    data = frame.head(5)
    rets = pl.DataFrame({"A": [0.01, 0.02, 0.00, -0.01, 0.03]})
    data = data.with_columns(pl.Series("A", rets["A"]))
    s = Stats(data)

    with pytest.raises(TypeError):
        # periods is a string, which should trigger the explicit type check
        s.volatility(periods="252")


def test_best_and_worst_handles_nulls_and_returns_extremes(frame):
    """best() returns the max and worst() returns the min, ignoring nulls."""
    data = frame.head(6)
    a = [None, -1.0, 0.0, 2.5, None, -3.0]
    b = [0.1, 0.2, 0.3, 0.0, -0.1, None]
    pl.DataFrame({"A": a, "B": b})
    data = data.with_columns(pl.Series("A", a), pl.Series("B", b))
    s = Stats(data)

    best = s.best()
    worst = s.worst()

    assert best == {"A": 2.5, "B": 0.3}
    assert worst == {"A": -3.0, "B": -0.1}


def test_value_at_risk_sigma_parameter_scales_tail(frame):
    """Increasing sigma should make VaR more negative (further into the tail)."""
    data = frame.head(200)

    base = np.linspace(-3, 3, num=200)
    a = (base / 100.0).tolist()
    data = data.with_columns(pl.Series("A", a))
    s = Stats(data)

    var1 = s.value_at_risk(alpha=0.05, sigma=1.0)["A"]
    var2 = s.value_at_risk(alpha=0.05, sigma=2.0)["A"]

    assert var2 < var1  # more negative
