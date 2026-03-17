"""Property-based tests for basanos.math.optimizer (BasanosEngine).

Uses Hypothesis to systematically explore structural invariants for the
correlation-aware optimizer across varied configurations and price data.

Properties under test
---------------------
ret_adj
  - Output shape equals input prices shape.
  - All non-null, non-NaN values lie within [-clip, +clip].

vola
  - All finite volatility values are non-negative.

position_leverage
  - All leverage values are non-negative (L1 norm >= 0).

condition_number
  - All finite condition numbers are >= 1.0 (max_sv / min_sv >= 1).

effective_rank
  - All finite effective-rank values lie in [1, n_assets].

solver_residual
  - All non-NaN residual values are non-negative (Euclidean norm >= 0).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_st

from basanos.math import BasanosConfig, BasanosEngine

# ─── Shared strategies ────────────────────────────────────────────────────────

_CLIP = st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False)
_SHRINK = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_AUM = st.floats(min_value=1e4, max_value=1e8, allow_nan=False, allow_infinity=False)


@st.composite
def _engine(draw: st.DrawFn) -> BasanosEngine:
    """Strategy for a valid BasanosEngine with small non-monotonic price data.

    Generates non-monotonic positive prices by alternating the sign of each
    percentage change, guaranteeing the series has both up and down moves.
    Config parameters (vola, corr, clip, shrink, aum) are drawn independently.
    The number of rows is set to corr + 5 or more so that EWM warm-up yields
    at least a few finite outputs.
    """
    n_assets = draw(st.integers(min_value=1, max_value=2))
    vola = draw(st.integers(min_value=3, max_value=8))
    corr = draw(st.integers(min_value=vola, max_value=vola + 10))
    # Enough rows for EWM warm-up (corr + 5 guarantees at least 5 finite outputs)
    n_rows = draw(st.integers(min_value=corr + 5, max_value=corr + 20))
    clip = draw(_CLIP)
    shrink = draw(_SHRINK)
    aum = draw(_AUM)
    cfg = BasanosConfig(vola=vola, corr=corr, clip=clip, shrink=shrink, aum=aum)

    dates = list(range(n_rows))
    price_cols: dict = {"date": pl.Series(dates, dtype=pl.Int64)}
    mu_cols: dict = {"date": pl.Series(dates, dtype=pl.Int64)}

    for i in range(n_assets):
        col = f"a{i}"

        # Draw percentage magnitudes for a random walk; alternate signs so the
        # price series is guaranteed to be non-monotonic (up-down-up-down...).
        pct = draw(
            np_st.arrays(
                dtype=np.float64,
                shape=n_rows - 1,
                elements=st.floats(min_value=0.005, max_value=0.05, allow_nan=False, allow_infinity=False),
            )
        )
        signs = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(n_rows - 1)])
        base = draw(st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False))

        # Build price series: each step multiplies previous price by (1 ± pct).
        # Alternating signs ensure the series is never strictly monotonic.
        prices = np.empty(n_rows)
        prices[0] = base
        for j in range(1, n_rows):
            prices[j] = prices[j - 1] * (1.0 + signs[j - 1] * pct[j - 1])

        price_cols[col] = pl.Series(prices.tolist(), dtype=pl.Float64)

        mu_arr = draw(
            np_st.arrays(
                dtype=np.float64,
                shape=n_rows,
                elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            )
        )
        mu_cols[col] = pl.Series(mu_arr.tolist(), dtype=pl.Float64)

    return BasanosEngine(
        prices=pl.DataFrame(price_cols),
        mu=pl.DataFrame(mu_cols),
        cfg=cfg,
    )


# ─── ret_adj ──────────────────────────────────────────────────────────────────


@given(engine=_engine())
@settings(max_examples=100)
def test_ret_adj_shape_matches_prices(engine: BasanosEngine) -> None:
    """ret_adj must have the same shape as the input prices DataFrame."""
    assert engine.ret_adj.shape == engine.prices.shape


@given(engine=_engine())
@settings(max_examples=100)
def test_ret_adj_within_clip_bounds(engine: BasanosEngine) -> None:
    """All non-null, non-NaN ret_adj values must lie within [-clip, +clip]."""
    clip = engine.cfg.clip
    tolerance = 1e-9
    for col in engine.assets:
        vals = [v for v in engine.ret_adj[col].drop_nulls().to_list() if not math.isnan(v)]
        out_of_bounds = [v for v in vals if abs(v) > clip + tolerance]
        assert not out_of_bounds, f"Values in '{col}' exceed clip={clip}: {out_of_bounds}"


# ─── vola ─────────────────────────────────────────────────────────────────────


@given(engine=_engine())
@settings(max_examples=100)
def test_vola_finite_values_are_non_negative(engine: BasanosEngine) -> None:
    """All finite EWMA volatility values must be >= 0."""
    for col in engine.assets:
        vals = [v for v in engine.vola[col].drop_nulls().to_list() if math.isfinite(v)]
        negative = [v for v in vals if v < 0.0]
        assert not negative, f"Negative vola in '{col}': {negative}"


# ─── position_leverage ────────────────────────────────────────────────────────


@given(engine=_engine())
@settings(max_examples=100)
def test_position_leverage_is_non_negative(engine: BasanosEngine) -> None:
    """All leverage values must be non-negative (L1 norm is always >= 0)."""
    leverage = engine.position_leverage["leverage"].to_list()
    negative = [v for v in leverage if v < 0.0]
    assert not negative, f"Negative leverage: {negative}"


# ─── condition_number ─────────────────────────────────────────────────────────


@given(engine=_engine())
@settings(max_examples=100)
def test_condition_number_at_least_one(engine: BasanosEngine) -> None:
    """All finite condition numbers must be >= 1.0 (max_sv / min_sv >= 1)."""
    kappas = [v for v in engine.condition_number["condition_number"].to_list() if v is not None and math.isfinite(v)]
    tolerance = 1e-9
    below_one = [v for v in kappas if v < 1.0 - tolerance]
    assert not below_one, f"Condition number below 1.0: {below_one}"


# ─── effective_rank ───────────────────────────────────────────────────────────


@given(engine=_engine())
@settings(max_examples=100)
def test_effective_rank_bounded_by_asset_count(engine: BasanosEngine) -> None:
    """All finite effective-rank values must lie in [1, n_assets]."""
    n_assets = len(engine.assets)
    ranks = [v for v in engine.effective_rank["effective_rank"].to_list() if v is not None and math.isfinite(v)]
    tolerance = 1e-9
    out_of_range = [v for v in ranks if not (1.0 - tolerance <= v <= n_assets + tolerance)]
    assert not out_of_range, f"Effective rank outside [1, {n_assets}]: {out_of_range}"


# ─── solver_residual ──────────────────────────────────────────────────────────


@given(engine=_engine())
@settings(max_examples=100)
def test_solver_residual_is_non_negative(engine: BasanosEngine) -> None:
    """All non-NaN solver residuals must be non-negative (Euclidean norm >= 0)."""
    residuals = [v for v in engine.solver_residual["residual"].to_list() if v is not None and math.isfinite(v)]
    negative = [v for v in residuals if v < 0.0]
    assert not negative, f"Negative residuals: {negative}"
