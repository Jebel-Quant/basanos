"""Additional edge-case tests for BasanosEngine to improve coverage.

This module targets branches in BasanosEngine.cash_position that are hard to hit
with regular random data, in particular the path where at time i>0 the current
row has finite prices but all returns for that row are NaN because the previous
row had only NaNs. This exercises the ``ret_mask.any() == False`` branch that
skips profit variance updates.
"""

from __future__ import annotations

import polars as pl

from basanos.math import BasanosConfig, BasanosEngine


def test_cash_position_skips_profit_update_when_returns_all_nan():
    """Ensure branch where ret_mask.any() is False is exercised.

    We construct prices where the first row has NaNs for all asset prices and the
    second row has finite prices. This makes the computed returns for row 1 all
    NaN (since they depend on the previous row), while the mask for row 1 is
    True (finite prices are present). The engine should skip the profit-variance
    update for that row without error.
    """
    dates = pl.date_range(start=pl.date(2020, 1, 1), end=pl.date(2020, 1, 5), interval="1d", eager=True)

    # Row 0 all-NaN asset prices; subsequent rows finite
    a = [None, 101.0, 102.0, 103.0, 104.0]
    b = [None, 201.0, 202.0, 203.0, 204.0]
    prices = pl.DataFrame({"date": dates, "A": a, "B": b}).with_columns(pl.col(["A", "B"]).cast(pl.Float64))

    # Simple bounded signals for mu; include date for alignment
    mu = pl.DataFrame(
        {
            "date": dates,
            "A": pl.Series([0.1, 0.2, 0.1, -0.1, 0.0], dtype=pl.Float64),
            "B": pl.Series([0.0, -0.2, -0.1, 0.2, 0.1], dtype=pl.Float64),
        }
    )

    # Set corr >= vola to satisfy validator; small values to ensure early availability
    cfg = BasanosConfig(vola=1, corr=2, clip=3.0, shrink=0.5, aum=1e6)

    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
    cp = engine.cash_position

    # Basic schema checks
    assert cp.columns[0] == "date"
    assert cp.columns[1:] == ["A", "B"]

    # Row 1 should have finite prices but returns computed from row 0 are NaN -> branch hit
    # We mainly assert that computation succeeds and produces a DataFrame with expected dtypes
    assert isinstance(cp, pl.DataFrame)
    # After warmup (corr=2), computation should succeed; values may still contain NaNs
    tail = cp.tail(cp.height - cfg.corr)
    for c in ("A", "B"):
        # Ensure the series exists and has floating dtype; NaNs are acceptable in this edge setup
        assert tail.schema[c] == pl.Float64
