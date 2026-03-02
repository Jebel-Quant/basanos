"""Additional tests to drive coverage of taipan.math._taipan to 100%.

These tests focus on constructor validation, helper properties (ret_adj, vola,
cor), and error paths not exercised by the existing portfolio tests.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from basanos.math import TaipanConfig, TaipanEngine


@pytest.fixture
def small_prices() -> pl.DataFrame:
    """Create a tiny deterministic prices frame with two assets.

    The series are monotonic to keep correlations well-defined.
    """
    n = 10
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    a = pl.Series([100.0 + i for i in range(n)], dtype=pl.Float64)
    b = pl.Series([200.0 + 2 * i for i in range(n)], dtype=pl.Float64)
    return pl.DataFrame({"date": dates, "A": a, "B": b})


@pytest.fixture
def small_mu(small_prices: pl.DataFrame) -> pl.DataFrame:
    """Create a small mu frame aligned with ``small_prices``.

    Uses a bounded signal to avoid extreme values.
    """
    n = small_prices.height
    theta = np.linspace(0.0, np.pi, num=n)
    mu_a = np.tanh(np.sin(theta))
    mu_b = np.tanh(np.cos(theta))
    return pl.DataFrame(
        {
            "date": small_prices["date"],
            "A": pl.Series(mu_a, dtype=pl.Float64),
            "B": pl.Series(mu_b, dtype=pl.Float64),
        }
    )


def test_post_init_shape_mismatch_raises(small_prices: pl.DataFrame, small_mu: pl.DataFrame) -> None:
    """Prices and mu must have identical shapes (rows and columns)."""
    cfg = TaipanConfig(vola=4, corr=4, clip=3.0, shrink=0.5, aum=1e6)
    # Drop last row from mu to force a shape mismatch
    mu_bad = small_mu.slice(0, small_mu.height - 1)
    with pytest.raises(ValueError, match=r".*"):
        _ = TaipanEngine(prices=small_prices, mu=mu_bad, cfg=cfg)


def test_post_init_missing_date_raises() -> None:
    """Both prices and mu must contain a 'date' column."""
    cfg = TaipanConfig(vola=4, corr=4, clip=3.0, shrink=0.5, aum=1e6)
    prices_no_date = pl.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    mu_ok = pl.DataFrame(
        {
            "date": pl.date_range(date(2020, 1, 1), date(2020, 1, 2), interval="1d", eager=True),
            "A": [0.1, 0.2],
            "B": [0.3, 0.4],
        }
    )
    with pytest.raises(ValueError, match=r".*"):
        _ = TaipanEngine(prices=prices_no_date, mu=mu_ok, cfg=cfg)

    prices_ok = pl.DataFrame(
        {
            "date": pl.date_range(date(2020, 1, 1), date(2020, 1, 2), interval="1d", eager=True),
            "A": [1.0, 2.0],
            "B": [3.0, 4.0],
        }
    )
    mu_no_date = pl.DataFrame({"A": [0.1, 0.2], "B": [0.3, 0.4]})
    with pytest.raises(ValueError, match=r".*"):
        _ = TaipanEngine(prices=prices_ok, mu=mu_no_date, cfg=cfg)


def test_post_init_column_mismatch_raises(small_prices: pl.DataFrame, small_mu: pl.DataFrame) -> None:
    """Prices and mu must have identical column sets."""
    cfg = TaipanConfig(vola=4, corr=4, clip=3.0, shrink=0.5, aum=1e6)
    mu_bad_cols = small_mu.rename({"B": "BB"})
    with pytest.raises(ValueError, match=r".*"):
        _ = TaipanEngine(prices=small_prices, mu=mu_bad_cols, cfg=cfg)


def test_ret_adj_and_vola_return_frames_with_asset_columns(small_prices: pl.DataFrame, small_mu: pl.DataFrame) -> None:
    """ret_adj and vola should return DataFrames aligned with asset columns."""
    cfg = TaipanConfig(vola=3, corr=4, clip=2.5, shrink=0.5, aum=1e6)
    engine = TaipanEngine(prices=small_prices, mu=small_mu, cfg=cfg)

    ra = engine.ret_adj
    vo = engine.vola

    assert isinstance(ra, pl.DataFrame)
    assert isinstance(vo, pl.DataFrame)
    # Ensure date column is preserved and asset columns are present
    assert ra.columns[0] == "date"
    assert vo.columns[0] == "date"
    assert set(ra.columns[1:]) == {"A", "B"}
    assert set(vo.columns[1:]) == {"A", "B"}


def test_cor_returns_square_matrices(small_prices: pl.DataFrame, small_mu: pl.DataFrame) -> None:
    """Cor should return a dict of square correlation matrices of size n_assets."""
    # Use a small corr window to ensure availability of finite correlations
    cfg = TaipanConfig(vola=2, corr=2, clip=2.5, shrink=0.5, aum=1e6)
    engine = TaipanEngine(prices=small_prices, mu=small_mu, cfg=cfg)

    cor = engine.cor
    assert isinstance(cor, dict)

    n_assets = len(engine.assets)
    # There should be at least one timestamp with a correlation matrix
    assert len(cor) > 0
    for _, mat in cor.items():
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (n_assets, n_assets)
    # We intentionally do not assert finiteness since early windows may produce NaNs


def test_cash_position_skips_rows_with_all_nan_prices() -> None:
    """cash_position should skip timestamps where all asset prices are non-finite.

    We craft prices with a single row containing only nulls for all assets after the
    warmup so that ``mask.any()`` is False and the ``continue`` branch is executed.
    The resulting cash positions at that timestamp should remain null (NaN).
    """
    n = 12
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)
    # Base increasing prices
    a = [100.0 + i for i in range(n)]
    b = [200.0 + 2 * i for i in range(n)]
    # Introduce an all-NaN row after warmup
    idx_nan = 6
    a[idx_nan] = None  # type: ignore[assignment]
    b[idx_nan] = None  # type: ignore[assignment]
    prices = pl.DataFrame({"date": dates, "A": a, "B": b}).with_columns(
        pl.col("A").cast(pl.Float64), pl.col("B").cast(pl.Float64)
    )

    # Simple bounded mu
    theta = np.linspace(0.0, np.pi, num=n)
    mu = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series(np.tanh(np.sin(theta)), dtype=pl.Float64),
            "B": pl.Series(np.tanh(np.cos(theta)), dtype=pl.Float64),
        }
    )

    cfg = TaipanConfig(vola=3, corr=4, clip=3.0, shrink=0.5, aum=1e6)
    engine = TaipanEngine(prices=prices, mu=mu, cfg=cfg)
    cp = engine.cash_position

    # The NaN row should produce null cash positions for both assets
    row = cp.row(idx_nan, named=True)
    assert row["A"] is None or (isinstance(row["A"], float) and np.isnan(row["A"]))
    assert row["B"] is None or (isinstance(row["B"], float) and np.isnan(row["B"]))
