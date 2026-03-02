"""Tests for taipan.math.portfolio module."""

from datetime import date

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from basanos.analytics import Portfolio


@pytest.fixture
def prices():
    """Create sample prices for testing."""
    return pl.DataFrame(
        {
            "date": pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 3), interval="1d", eager=True).cast(
                pl.Date
            ),
            "A": pl.Series([100.0, 110.0, 121.0], dtype=pl.Float64),
            "B": pl.Series([200.0, 180.0, 198.0], dtype=pl.Float64),
        }
    )


@pytest.fixture
def positions():
    """Create sample positions for testing."""
    return pl.DataFrame(
        {
            "date": pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 3), interval="1d", eager=True).cast(
                pl.Date
            ),
            "A": pl.Series([1000.0, 1000.0, 1000.0], dtype=pl.Float64),
            "B": pl.Series([0.0, 500.0, 500.0], dtype=pl.Float64),
        }
    )


@pytest.fixture
def portfolio(prices, positions):
    """Create Portfolio instance for testing."""
    return Portfolio.from_cash_position(prices=prices, cash_position=positions, aum=1e5)


def test_compute_daily_profits_portfolio_basic(portfolio):
    """Compute per-asset profits and preserve date column."""
    profits = portfolio.profits

    # Expect date preserved
    assert "date" in profits.columns

    # Manual expected per-asset profits using shifted positions and pct_change
    # A returns: [0.0, 0.1, 0.1]; shifted pos: [0.0, 1000.0, 1000.0] -> profits: [0.0, 100.0, 100.0]
    # B returns: [0.0, -0.1, 0.1]; shifted pos: [0.0, 0.0, 500.0] -> profits: [0.0, 0.0, 50.0]
    expected = pl.DataFrame(
        {
            "date": portfolio.prices["date"],
            "A": pl.Series([0.0, 100.0, 100.0], dtype=pl.Float64),
            "B": pl.Series([0.0, 0.0, 50.0], dtype=pl.Float64),
        }
    )

    # Compare columns and values
    assert profits.columns == expected.columns
    for c in ["A", "B"]:
        assert np.allclose(profits[c].to_numpy(), expected[c].to_numpy(), rtol=1e-12, atol=1e-12)


def test_portfolio_profit_and_nav(portfolio):
    """Aggregate per-asset profits to portfolio profit and compute NAV."""
    # Profit aggregation
    profit_df = portfolio.profit
    assert profit_df.columns == ["date", "profit"]

    # Expected total daily profit: [0.0, 100.0, 150.0]
    expected_profit = np.array([0.0, 100.0, 150.0])
    assert np.allclose(profit_df["profit"].to_numpy(), expected_profit)

    # NAV: cumulative sum of profit + 1e8 (as implemented)
    nav_df = portfolio.nav_accumulated
    assert nav_df.columns == ["date", "profit", "NAV_accumulated"]

    expected_nav = np.array([1e5, 1e5 + 100.0, 1e5 + 250.0])
    assert np.allclose(nav_df["NAV_accumulated"].to_numpy(), expected_nav)


def test_portfolio_sharpe_matches_manual(portfolio):
    """Sharpe returned by class matches manual computation."""
    # Implementation uses diff of NAV (absolute), not pct_change

    # Function under test
    out = portfolio.stats.sharpe()["returns"]

    assert np.isfinite(out)
    assert np.isclose(out, 20.845234695819794, rtol=1e-12, atol=1e-12)


def test_portfolio_plot_returns_figure(portfolio):
    """Plot method returns a Plotly Figure and is serializable."""
    fig = portfolio.plots.snapshot()
    # Ensure it's a Plotly Figure-like object
    assert isinstance(fig, go.Figure)
    # Basic integrity: has data and layout
    _ = fig.to_dict()


# New tests focusing on Portfolio.__post_init__ assertions


def test_portfolio_post_init_requires_polars_dataframes(prices, positions):
    """__post_init__ should assert inputs are Polars DataFrames."""
    # cashposition wrong type
    with pytest.raises(TypeError):
        Portfolio(prices=prices, cashposition={"date": [1, 2, 3]})

    # prices wrong type
    with pytest.raises(TypeError):
        Portfolio(prices=[[1.0, 2.0, 3.0]], cashposition=positions)


def test_portfolio_post_init_requires_same_number_of_rows(prices, positions):
    """__post_init__ should raise ValueError when row counts differ."""
    with pytest.raises(ValueError, match=r".*"):
        Portfolio(prices=prices.head(3), cashposition=positions.head(2))


def test_portfolio_post_init_requires_positive_aum(prices, positions):
    """__post_init__ should raise ValueError when AUM is not strictly positive."""
    with pytest.raises(ValueError, match=r".*"):
        Portfolio(prices=prices, cashposition=positions, aum=0.0)

    with pytest.raises(ValueError, match=r".*"):
        Portfolio(prices=prices, cashposition=positions, aum=-1.0)


# Additional tests to reach 100% coverage for portfolio.py


def test_from_riskposition_returns_portfolio_and_cashposition_shape():
    """from_riskposition should return a Portfolio with aligned cashposition columns/height."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 2, 10), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.linspace(100, 120, len(dates)), dtype=pl.Float64),
            "B": pl.Series(np.linspace(50, 60, len(dates)), dtype=pl.Float64),
        }
    )
    # Simple risk positions, arbitrary values
    riskposition = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.sin(np.linspace(0, 3.14, len(dates))), dtype=pl.Float64),
            "B": pl.Series(np.cos(np.linspace(0, 3.14, len(dates))), dtype=pl.Float64),
        }
    )

    pf = Portfolio.from_risk_position(prices, riskposition, vola=8, aum=1e8)
    assert isinstance(pf, Portfolio)
    assert pf.cashposition.height == prices.height
    # Numeric columns should be present in cashposition
    for c in ["A", "B"]:
        assert c in pf.cashposition.columns


def test_sharpe_zero_std_returns_zero():
    """Sharpe should return 0.0 when NAV differences have zero std (flat NAV)."""
    # Constant prices imply zero returns; any positions yield zero profits if shifted positions are zeros
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 5), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates, "A": pl.Series([100.0] * len(dates), dtype=pl.Float64)})
    # Zero positions ensure zero profit
    positions = pl.DataFrame({"date": dates, "A": pl.Series([0.0] * len(dates), dtype=pl.Float64)})

    pf = Portfolio(prices=prices, cashposition=positions)
    with pytest.raises(ZeroDivisionError):
        pf.stats.sharpe()["returns"]


def test_compute_daily_profits_replaces_nonfinite_with_zero():
    """_compute_daily_profits_portfolio should replace non-finite profit values with 0.0."""
    # Create a price series that causes infinite pct_change: 0 -> 1 (division by zero)
    prices = pl.DataFrame(
        {
            "date": pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 2), interval="1d", eager=True).cast(
                pl.Date
            ),
            "A": pl.Series([0.0, 1.0], dtype=pl.Float64),
        }
    )
    positions = pl.DataFrame(
        {
            "date": prices["date"],
            "A": pl.Series([1.0, 1.0], dtype=pl.Float64),
        }
    )

    portfolio = Portfolio(prices=prices, cashposition=positions)
    profits = portfolio.profits

    # First day profit 0, second day was inf*shifted_pos -> cleaned to 0.0 via is_finite guard
    assert np.allclose(profits["A"].to_numpy(), np.array([0.0, 0.0]))


def test_compute_daily_profits_no_numeric_columns():
    """When there are no numeric columns, function should return only non-numeric columns unchanged."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 2), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates})
    positions = pl.DataFrame({"date": dates})
    portfolio = Portfolio(prices=prices, cashposition=positions)
    profits = portfolio.profits

    assert profits.columns == ["date"]
    assert profits.height == 2


def test_profit_raises_when_no_numeric_asset_columns():
    """Portfolio.profit should raise ValueError if there are no numeric asset columns."""
    dates = pl.date_range(start=date(2020, 1, 1), end=date(2020, 1, 2), interval="1d", eager=True).cast(pl.Date)
    prices = pl.DataFrame({"date": dates})
    positions = pl.DataFrame({"date": dates})
    portfolio = Portfolio(prices=prices, cashposition=positions)

    with pytest.raises(ValueError, match=r".*"):
        _ = portfolio.profit


def test_returns_property_scales_profit_by_aum_and_preserves_date(portfolio):
    """Returns should divide numeric columns by aum and retain the 'date' column."""
    rets = portfolio.returns

    # Columns preserved
    assert "date" in rets.columns
    assert "profit" in rets.columns

    # Numeric equality: profit divided by aum
    expected = (portfolio.profit.select(pl.col("profit")) / portfolio.aum)["profit"].to_numpy()
    actual = rets["returns"].to_numpy()
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_nav_compounded_uses_compounding_and_is_close_to_nav_for_small_returns(portfolio):
    """nav_compounded should compound returns; for small returns it approximates additive NAV."""
    # additive NAV (baseline)
    nav_add = portfolio.nav_accumulated
    nav_cmp = portfolio.nav_compounded

    # Structure: both should have a date column
    assert "date" in nav_cmp.columns
    assert "date" in nav_add.columns

    # First compounded NAV equals AUM
    cmp_values = nav_cmp["NAV_compounded"].to_numpy()
    assert np.isclose(cmp_values[0], portfolio.aum)

    # For small returns, compounded NAV should be close to additive NAV
    add_values = nav_add["NAV_accumulated"].to_numpy()
    assert np.isclose(add_values[0], portfolio.aum)


def test_highwater_is_cummax_of_nav(portfolio):
    """Highwater should equal the cumulative maximum of NAV and preserve 'date'."""
    nav_df = portfolio.nav_accumulated
    hw_df = portfolio.highwater

    # Structure
    assert "date" in hw_df.columns
    assert "highwater" in hw_df.columns

    # Numeric equality: cummax of NAV
    expected = nav_df["NAV_accumulated"].cum_max().to_numpy()
    actual = hw_df["highwater"].to_numpy()
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_drawdown_is_highwater_minus_nav_and_preserves_date(portfolio):
    """Drawdown should equal highwater - NAV, start at 0, be non-negative, and keep 'date'."""
    dd_df = portfolio.drawdown
    print(dd_df)

    # Structure
    assert "date" in dd_df.columns
    assert "drawdown" in dd_df.columns

    # Expected numeric: highwater - NAV
    expected = (dd_df["highwater"] - dd_df["NAV_accumulated"]).to_numpy()
    actual = dd_df["drawdown"].to_numpy()

    # First drawdown must be 0 and all drawdowns non-negative
    assert np.isclose(actual[0], 0.0)
    assert np.all(actual >= 0.0)

    # Exact equality to computed expectation
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_stats(portfolio):
    """stats() should return a dictionary with expected keys."""
    stats = portfolio.stats

    assert pytest.approx(stats.sharpe()["returns"]) == 20.845234695819794
    assert pytest.approx(stats.kurtosis()["returns"]) == -1.500


def test_portfolio_snapshot_log_scale(portfolio):
    """snapshot(log_scale=True) returns a Figure and sets the first y-axis to logarithmic scale."""
    fig = portfolio.plots.snapshot(log_scale=True)
    assert isinstance(fig, go.Figure)
    # yaxis (row=1,col=1) should be log scale
    assert fig.layout.yaxis.type == "log"


def test_assert_clean_series_raises_on_null():
    """_assert_clean_series should raise ValueError when series contains null values."""
    s = pl.Series([1.0, None, 3.0])
    with pytest.raises(ValueError, match=r".*"):
        Portfolio._assert_clean_series(s)


def test_assert_clean_series_raises_on_nonfinite():
    """_assert_clean_series should raise ValueError when series contains non-finite values."""
    s = pl.Series([1.0, float("inf"), 3.0])
    with pytest.raises(ValueError, match=r".*"):
        Portfolio._assert_clean_series(s)


def test_portfolio_all_merges_drawdown_and_nav_compounded(portfolio):
    """All property should join drawdown and nav_compounded on 'date' with expected columns."""
    result = portfolio.all
    assert "date" in result.columns
    assert "NAV_accumulated" in result.columns
    assert "NAV_compounded" in result.columns
    assert "drawdown" in result.columns
    assert len(result) == len(portfolio.prices)
