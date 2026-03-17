"""Tests for basanos.analytics._stats (Stats class)."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from basanos.analytics._stats import Stats, _to_float, _to_float_or_none

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def dates() -> list[datetime]:
    """Return a list of consecutive daily datetimes used as a test index."""
    dt0 = datetime.fromisoformat("2020-01-01")
    return [dt0 + timedelta(days=i) for i in range(200)]


@pytest.fixture
def frame(dates: list[datetime]) -> pl.DataFrame:
    """Build a Polars DataFrame with a 'date' column from the dates fixture."""
    return pl.DataFrame({"date": dates})


def _make_daily_frame(days: int = 10) -> pl.DataFrame:
    """Build a simple DataFrame with a daily 'date' column and one numeric asset."""
    start = date(2020, 1, 1)
    end = start + timedelta(days=days - 1)
    dates = pl.date_range(start=start, end=end, interval="1d", eager=True)
    a = pl.Series("A", [float(i) for i in range(days)], dtype=pl.Float64)
    return pl.DataFrame({"date": dates, "A": a, "label": ["x"] * days})


# ─── Initialisation ───────────────────────────────────────────────────────────


def test_stats_init_raises_typeerror_when_not_dataframe():
    """Passing a non-DataFrame to Stats should raise TypeError in __post_init__."""
    with pytest.raises(TypeError):
        _ = Stats(data={"date": [1, 2, 3], "A": [0.1, 0.2, 0.3]})


def test_stats_init_raises_valueerror_on_empty_dataframe():
    """Passing an empty DataFrame should raise ValueError in __post_init__."""
    empty_df = pl.DataFrame(
        {
            "date": pl.Series("date", [], dtype=pl.Datetime("ns")),
            "A": pl.Series("A", [], dtype=pl.Float64),
        }
    )
    with pytest.raises(ValueError, match=r".*"):
        _ = Stats(data=empty_df)


# ─── Assets property and periods_per_year ────────────────────────────────────


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


def test_periods_per_year_single_row_uses_fallback():
    """Single-row frame: diff is empty → mean is None → fallback to 86400 s → 365 days/year."""
    df = pl.DataFrame(
        {
            "date": pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 1), interval="1d", eager=True),
            "A": [1.0],
        }
    )
    s = Stats(df)
    assert math.isclose(s.periods_per_year, 365.0, rel_tol=1e-12)


def test_periods_per_year_numeric_first_column_uses_to_float():
    """Numeric first column: mean_diff is a float → elif branch executed."""
    df = pl.DataFrame({"steps": [1, 2, 3], "A": [1.0, 2.0, 3.0]})
    s = Stats(df)
    expected = 365.0 * 24.0 * 3600.0
    assert math.isclose(s.periods_per_year, expected, rel_tol=1e-12)


# ─── Private helper functions ─────────────────────────────────────────────────


def test_to_float_none_returns_zero():
    """_to_float(None) should return 0.0, treating absence as a zero contribution."""
    assert _to_float(None) == 0.0


def test_to_float_timedelta_returns_total_seconds():
    """_to_float(timedelta) should return total_seconds() as float."""
    assert _to_float(timedelta(seconds=42)) == 42.0


def test_to_float_or_none_none_returns_none():
    """_to_float_or_none(None) should propagate None rather than coercing to zero.

    This contrasts with _to_float which converts None to 0.0; the *_or_none
    variant preserves the absence signal for callers that need to distinguish
    missing from zero.
    """
    assert _to_float_or_none(None) is None


def test_to_float_or_none_timedelta_returns_total_seconds():
    """_to_float_or_none(timedelta) should return total_seconds() as float."""
    assert _to_float_or_none(timedelta(seconds=7)) == 7.0


# ─── Skew ─────────────────────────────────────────────────────────────────────


def test_skew_symmetric_zero(frame):
    """Symmetric distribution around zero should have skew approximately zero."""
    data = frame.head(8).with_columns(pl.Series("A", [-3.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 3.0]))
    s = Stats(data)
    result = s.skew()
    assert set(result.keys()) == {"A"}
    assert math.isclose(result["A"], 0.0, abs_tol=1e-12, rel_tol=1e-9)


def test_skew_positive_and_negative_signs(frame):
    """Positively skewed data yields positive skew; negatively skewed yields negative skew."""
    data = frame.head(6).with_columns(
        pl.Series("A", [-0.5, -0.2, -0.1, 0.0, 0.0, 3.0]),
        pl.Series("B", [0.5, 0.2, 0.1, 0.0, 0.0, -3.0]),
    )
    out = Stats(data).skew()
    assert out["A"] > 0
    assert out["B"] < 0
    assert math.isclose(abs(out["A"]), abs(out["B"]), rel_tol=0.25)


def test_skew_ignores_nulls_and_returns_dict(frame):
    """Skew should ignore nulls and return a mapping of asset->float values."""
    data = frame.head(7).with_columns(pl.Series("A", [None, 0.0, 0.1, None, -0.1, 0.2, -0.2]))
    res: dict[str, Any] = Stats(data).skew()
    assert isinstance(res, dict)
    assert set(res) == {"A"}
    assert isinstance(res["A"], float)
    assert math.isfinite(res["A"])
    assert res["A"] == pytest.approx(0.0, abs=1e-12)


# ─── Kurtosis ─────────────────────────────────────────────────────────────────


def test_kurtosis_heavy_and_flat_signs(frame):
    """Kurtosis should be positive for heavy tails and negative for flat/uniform-like data."""
    data = frame.head(10).with_columns(
        pl.Series("A", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, -10.0, 0.0]),
        pl.Series("B", [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    out = Stats(data).kurtosis()
    assert out["A"] > 0
    assert out["B"] < 0


def test_kurtosis_ignores_nulls_and_returns_dict(frame):
    """Kurtosis should ignore nulls and return a mapping of asset->float values."""
    data = frame.head(8).with_columns(pl.Series("A", [None, -1.0, 0.0, 1.0, None, -2.0, 2.0, 0.0]))
    res = Stats(data).kurtosis()
    assert isinstance(res, dict)
    assert set(res.keys()) == {"A"}
    assert isinstance(res["A"], float)
    assert math.isfinite(res["A"])


# ─── Returns (avg / win / loss) ───────────────────────────────────────────────


def test_avg_return_win_loss(frame):
    """avg_return ignores zeros/nulls; avg_win/loss compute means of positive/negative subsets."""
    a = [0.1, 0.0, -0.2, None, 0.3, -0.4]
    data = frame.head(6).with_columns(pl.Series("A", a))
    rets = pl.DataFrame({"A": a})
    s = Stats(data)

    assert s.avg_return()["A"] == rets["A"].filter(rets["A"].is_not_null() & (rets["A"] != 0)).mean()
    assert s.avg_win()["A"] == rets["A"].filter(rets["A"] > 0).mean()
    assert s.avg_loss()["A"] == rets["A"].filter(rets["A"] < 0).mean()


# ─── Volatility and Sharpe ────────────────────────────────────────────────────


def test_volatility_paths(frame):
    """Volatility should annualize by sqrt(periods) when requested and accept custom periods."""
    a = [0.01, 0.02, 0.00, -0.01, 0.03]
    data = frame.head(5).with_columns(pl.Series("A", a))
    base_std = float(pl.DataFrame({"A": a})["A"].std())
    s = Stats(data)

    assert math.isclose(s.volatility(annualize=False)["A"], base_std, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(
        s.volatility(periods=252, annualize=True)["A"], base_std * math.sqrt(252), rel_tol=1e-12, abs_tol=1e-12
    )
    assert math.isclose(
        s.volatility(periods=60, annualize=True)["A"], base_std * math.sqrt(60), rel_tol=1e-12, abs_tol=1e-12
    )


def test_volatility_uses_default_periods_per_year_when_none():
    """volatility() without periods should scale by sqrt(periods_per_year)."""
    df = _make_daily_frame(12)
    s = Stats(df)
    base_std = float(df["A"].std())
    expected = base_std * math.sqrt(s.periods_per_year)
    assert abs(s.volatility()["A"] - expected) < 1e-12


def test_volatility_invalid_periods_raises_typeerror(frame):
    """Passing a non-numeric periods value to volatility should raise TypeError."""
    data = frame.head(5).with_columns(pl.Series("A", [0.01, 0.02, 0.00, -0.01, 0.03]))
    with pytest.raises(TypeError):
        Stats(data).volatility(periods="252")


def test_sharpe_simple(frame):
    """Sharpe equals mean/std (ddof=1) scaled by sqrt(periods)."""
    a = [0.01, 0.02, 0.00, -0.01, 0.03]
    data = frame.head(5).with_columns(pl.Series("A", a))
    rets = pl.DataFrame({"A": a})
    mean = float(rets["A"].mean())
    std_sample = float(rets["A"].std(ddof=1))
    expected = mean / std_sample

    out = Stats(data).sharpe(periods=1)
    assert math.isclose(out["A"], expected, rel_tol=1e-12, abs_tol=1e-12)


def test_sharpe_zero_volatility_returns_nan(frame):
    """Sharpe should return NaN when the series has zero volatility (constant returns)."""
    data = frame.head(5).with_columns(pl.Series("A", [0.01, 0.01, 0.01, 0.01, 0.01]))
    result = Stats(data).sharpe(periods=1)
    assert math.isnan(result["A"])


# ─── Risk (VaR / CVaR) ────────────────────────────────────────────────────────


def test_var_and_cvar_properties(frame):
    """VaR(alpha) should be more negative than mean; CVaR should be <= VaR for normal-like data."""
    a = (np.linspace(-3, 3, num=200) / 100.0).tolist()
    data = frame.head(200).with_columns(pl.Series("A", a))
    s = Stats(data)

    var = s.value_at_risk(alpha=0.05)["A"]
    cvar = s.conditional_value_at_risk(alpha=0.05)["A"]
    assert var < 0
    assert cvar <= var


def test_value_at_risk_sigma_parameter_scales_tail(frame):
    """Increasing sigma should make VaR more negative (further into the tail)."""
    a = (np.linspace(-3, 3, num=200) / 100.0).tolist()
    data = frame.head(200).with_columns(pl.Series("A", a))
    s = Stats(data)

    var1 = s.value_at_risk(alpha=0.05, sigma=1.0)["A"]
    var2 = s.value_at_risk(alpha=0.05, sigma=2.0)["A"]
    assert var2 < var1


# ─── Win rate / Profit factor / Payoff ratio ──────────────────────────────────


def test_win_rate_basic(frame):
    """win_rate returns fraction of strictly positive periods, ignoring nulls."""
    a = [0.1, -0.2, 0.0, 0.3, None, -0.1, 0.05]
    data = frame.head(7).with_columns(pl.Series("A", a))
    result = Stats(data).win_rate()
    # Positive: 0.1, 0.3, 0.05  → 3 out of 6 non-null → 0.5
    assert math.isclose(result["A"], 3 / 6, rel_tol=1e-12)


def test_win_rate_all_positive(frame):
    """win_rate returns 1.0 when all returns are positive."""
    data = frame.head(4).with_columns(pl.Series("A", [0.01, 0.02, 0.03, 0.04]))
    assert Stats(data).win_rate()["A"] == 1.0


def test_win_rate_all_negative(frame):
    """win_rate returns 0.0 when all returns are negative."""
    data = frame.head(4).with_columns(pl.Series("A", [-0.01, -0.02, -0.03, -0.04]))
    assert Stats(data).win_rate()["A"] == 0.0


def test_win_rate_all_nulls_returns_nan(frame):
    """win_rate returns NaN when the series is entirely null."""
    data = frame.head(3).with_columns(pl.Series("A", [None, None, None], dtype=pl.Float64))
    assert math.isnan(Stats(data).win_rate()["A"])


def test_profit_factor_basic(frame):
    """profit_factor equals sum(positive) / abs(sum(negative))."""
    a = [0.2, 0.1, -0.1, -0.05, 0.0]
    data = frame.head(5).with_columns(pl.Series("A", a))
    expected = 0.3 / 0.15
    assert math.isclose(Stats(data).profit_factor()["A"], expected, rel_tol=1e-12)


def test_profit_factor_no_losses_returns_inf(frame):
    """profit_factor returns inf when there are no losing periods."""
    data = frame.head(4).with_columns(pl.Series("A", [0.1, 0.2, 0.0, 0.3]))
    assert math.isinf(Stats(data).profit_factor()["A"])


def test_profit_factor_no_wins_returns_zero(frame):
    """profit_factor returns 0.0 when there are no winning periods."""
    data = frame.head(4).with_columns(pl.Series("A", [-0.1, -0.2, 0.0, -0.3]))
    assert Stats(data).profit_factor()["A"] == pytest.approx(0.0)


def test_payoff_ratio_basic(frame):
    """payoff_ratio equals mean(positive) / abs(mean(negative))."""
    a = [0.2, 0.4, -0.1, -0.3]
    data = frame.head(4).with_columns(pl.Series("A", a))
    avg_w = (0.2 + 0.4) / 2  # 0.3
    avg_l = (-0.1 + -0.3) / 2  # -0.2
    expected = avg_w / abs(avg_l)  # 1.5
    assert math.isclose(Stats(data).payoff_ratio()["A"], expected, rel_tol=1e-12)


def test_payoff_ratio_no_losses_returns_nan(frame):
    """payoff_ratio returns nan when there are no losing periods."""
    data = frame.head(3).with_columns(pl.Series("A", [0.1, 0.2, 0.3]))
    assert math.isnan(Stats(data).payoff_ratio()["A"])


# ─── Monthly win rate ──────────────────────────────────────────────────────────


def test_monthly_win_rate_basic():
    """monthly_win_rate returns fraction of months with positive compounded return."""
    # 3 months: Jan +, Feb -, Mar +  → 2/3
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 3, 31), interval="1d", eager=True)
    # Small positive return for Jan and Mar, small negative for Feb
    returns = []
    for d in dates.to_list():
        if d.month == 2:
            returns.append(-0.001)
        else:
            returns.append(0.001)
    df = pl.DataFrame({"date": dates, "A": returns})
    result = Stats(df).monthly_win_rate()
    assert math.isclose(result["A"], 2 / 3, rel_tol=1e-9)


def test_monthly_win_rate_no_date_column_returns_nan():
    """monthly_win_rate returns nan for each asset when there is no date column."""
    df = pl.DataFrame({"steps": [1, 2, 3], "A": [0.01, 0.02, -0.01]})
    result = Stats(df).monthly_win_rate()
    assert math.isnan(result["A"])


# ─── Worst N periods ──────────────────────────────────────────────────────────


def test_worst_n_periods_returns_n_smallest(frame):
    """worst_n_periods returns the N smallest values sorted ascending."""
    a = [0.05, -0.03, 0.10, -0.08, 0.02, -0.01]
    data = frame.head(6).with_columns(pl.Series("A", a))
    result = Stats(data).worst_n_periods(n=3)
    assert result["A"] == pytest.approx([-0.08, -0.03, -0.01], rel=1e-12)


def test_worst_n_periods_pads_with_none_when_fewer_observations(frame):
    """worst_n_periods pads list with None when series has fewer than n values."""
    data = frame.head(2).with_columns(pl.Series("A", [-0.1, 0.2]))
    result = Stats(data).worst_n_periods(n=5)
    assert result["A"][0] == pytest.approx(-0.1)
    assert result["A"][1] == pytest.approx(0.2)
    assert result["A"][2] is None
    assert len(result["A"]) == 5


def test_worst_n_periods_default_is_5(frame):
    """worst_n_periods defaults to returning 5 periods."""
    a = list(range(-10, 10, 1))
    data = frame.head(20).with_columns(pl.Series("A", [float(x) / 100 for x in a]))
    result = Stats(data).worst_n_periods()
    assert len(result["A"]) == 5
    assert result["A"][0] <= result["A"][1]  # sorted ascending


# ─── Up / Down capture ratio ──────────────────────────────────────────────────


def test_up_capture_ratio_basic(frame):
    """up_capture > 1 when strategy beats benchmark in up periods."""
    benchmark = pl.Series("bench", [0.01, 0.02, -0.01, 0.03, -0.005])
    # Strategy returns 2x benchmark on up days
    strategy = [0.02, 0.04, -0.005, 0.06, -0.002]
    data = frame.head(5).with_columns(pl.Series("A", strategy))
    result = Stats(data).up_capture(benchmark)
    assert result["A"] > 1.0


def test_down_capture_ratio_basic(frame):
    """down_capture < 1 when strategy loses less than benchmark in down periods."""
    benchmark = pl.Series("bench", [0.01, -0.02, -0.03, 0.04, -0.01])
    # Strategy loses half as much on down days
    strategy = [0.005, -0.01, -0.015, 0.02, -0.005]
    data = frame.head(5).with_columns(pl.Series("A", strategy))
    result = Stats(data).down_capture(benchmark)
    assert result["A"] < 1.0


def test_up_capture_no_up_periods_returns_nan(frame):
    """up_capture returns nan when benchmark never has positive periods."""
    benchmark = pl.Series("bench", [-0.01, -0.02, -0.03])
    data = frame.head(3).with_columns(pl.Series("A", [0.01, -0.01, 0.02]))
    result = Stats(data).up_capture(benchmark)
    assert math.isnan(result["A"])


def test_down_capture_no_down_periods_returns_nan(frame):
    """down_capture returns nan when benchmark never has negative periods."""
    benchmark = pl.Series("bench", [0.01, 0.02, 0.03])
    data = frame.head(3).with_columns(pl.Series("A", [0.01, -0.01, 0.02]))
    result = Stats(data).down_capture(benchmark)
    assert math.isnan(result["A"])


# ─── Best / Worst ─────────────────────────────────────────────────────────────


def test_best_and_worst_handles_nulls_and_returns_extremes(frame):
    """best() returns the max and worst() returns the min, ignoring nulls."""
    data = frame.head(6).with_columns(
        pl.Series("A", [None, -1.0, 0.0, 2.5, None, -3.0]),
        pl.Series("B", [0.1, 0.2, 0.3, 0.0, -0.1, None]),
    )
    s = Stats(data)
    assert s.best() == {"A": 2.5, "B": 0.3}
    assert s.worst() == {"A": -3.0, "B": -0.1}


# ─── Summary ──────────────────────────────────────────────────────────────────


def test_summary_returns_polars_dataframe(frame):
    """summary() should return a Polars DataFrame rather than a dict or other type.

    Callers that pass the summary directly to Polars/Pandas pipelines depend on
    this type contract.
    """
    data = frame.head(20).with_columns(pl.Series("A", [0.01 * i for i in range(1, 21)]))
    assert isinstance(Stats(data).summary(), pl.DataFrame)


def test_summary_has_metric_column_and_asset_columns(frame):
    """summary() should have a 'metric' column and one column per asset."""
    data = frame.head(20).with_columns(
        pl.Series("A", [0.01 * i for i in range(1, 21)]),
        pl.Series("B", [-0.01 * i for i in range(1, 21)]),
    )
    result = Stats(data).summary()
    assert "metric" in result.columns
    assert "A" in result.columns
    assert "B" in result.columns


def test_summary_metric_names(frame):
    """summary() should include expected metric names as rows."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 20), interval="1d", eager=True)
    data = pl.DataFrame({"date": dates, "A": [0.01 * i for i in range(1, 21)]})
    metric_names = Stats(data).summary()["metric"].to_list()
    expected = {
        "avg_return",
        "avg_win",
        "avg_loss",
        "win_rate",
        "profit_factor",
        "payoff_ratio",
        "monthly_win_rate",
        "best",
        "worst",
        "volatility",
        "sharpe",
        "skew",
        "kurtosis",
        "value_at_risk",
        "conditional_value_at_risk",
    }
    assert set(metric_names) == expected


def test_summary_values_match_individual_methods(frame):
    """summary() values should match what individual stat methods return."""
    a = [0.01 * i - 0.1 for i in range(1, 21)]
    data = frame.head(20).with_columns(pl.Series("A", a))
    s = Stats(data)
    result = s.summary()
    metric_map = dict(zip(result["metric"].to_list(), result["A"].to_list(), strict=False))

    assert metric_map["skew"] == pytest.approx(s.skew()["A"], rel=1e-9)
    assert metric_map["volatility"] == pytest.approx(s.volatility()["A"], rel=1e-9)
    assert metric_map["best"] == pytest.approx(s.best()["A"], rel=1e-9)
    assert metric_map["worst"] == pytest.approx(s.worst()["A"], rel=1e-9)
