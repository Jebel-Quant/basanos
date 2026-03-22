"""Tests for basanos.math._stream._StreamState.

Covers:
- Instantiation with correct array shapes.
- Mutability: fields can be reassigned after construction.
- Not frozen: dataclasses.replace is not required; direct attribute mutation works.
- Field shapes and dtypes match the documented contract.
- Private: _StreamState is not exported from basanos.math.

Tests for basanos.math._stream.StepResult.

Covers:
- Direct construction and field access
- Frozen (immutable) contract
- Public export from basanos.math
- BasanosStream.from_warmup() acceptance criterion
- BasanosStream.step() correctness and error handling

"""

from __future__ import annotations

import dataclasses
import time
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from basanos.exceptions import MissingDateColumnError
from basanos.math import (
    BasanosConfig,
    BasanosEngine,
    BasanosStream,
    SlidingWindowConfig,
    StepResult,
    WarmupState,
)
from basanos.math._stream import StepResult as StepResultDirect

# _StreamState, _ewm_std_from_state, and _ewm_vol_accumulators_from_batch are
# imported from the private module to enable isolation testing of the state
# extraction logic independently of the public API.
from basanos.math._stream import _ewm_std_from_state, _ewm_vol_accumulators_from_batch, _StreamState

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_state(n: int = 3) -> _StreamState:
    """Return a zero-initialised _StreamState for *n* assets."""
    return _StreamState(
        corr_zi_x=np.zeros((1, n, n)),
        corr_zi_x2=np.zeros((1, n, n)),
        corr_zi_xy=np.zeros((1, n, n)),
        corr_zi_w=np.zeros((1, n, n)),
        corr_count=np.zeros((n, n), dtype=int),
        vola_s_x=np.zeros(n),
        vola_s_x2=np.zeros(n),
        vola_s_w=np.zeros(n),
        vola_s_w2=np.zeros(n),
        vola_count=np.zeros(n, dtype=int),
        pct_s_x=np.zeros(n),
        pct_s_x2=np.zeros(n),
        pct_s_w=np.zeros(n),
        pct_s_w2=np.zeros(n),
        pct_count=np.zeros(n, dtype=int),
        prev_price=np.zeros(n),
        prev_cash_pos=np.zeros(n),
        step_count=0,
    )


# ─── Shape / dtype contract ───────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 3, 10])
def test_corr_zi_shapes(n: int) -> None:
    """corr_zi_* arrays must have shape (1, N, N)."""
    s = _make_state(n)
    for attr in ("corr_zi_x", "corr_zi_x2", "corr_zi_xy", "corr_zi_w"):
        arr = getattr(s, attr)
        assert arr.shape == (1, n, n), f"{attr}.shape expected (1,{n},{n}), got {arr.shape}"


@pytest.mark.parametrize("n", [1, 3, 10])
def test_corr_count_shape(n: int) -> None:
    """corr_count must have shape (N, N) with integer dtype."""
    s = _make_state(n)
    assert s.corr_count.shape == (n, n)
    assert np.issubdtype(s.corr_count.dtype, np.integer)


@pytest.mark.parametrize("n", [1, 3, 10])
def test_vola_accumulator_shapes(n: int) -> None:
    """vola_s_* and vola_count must have shape (N,)."""
    s = _make_state(n)
    for attr in ("vola_s_x", "vola_s_x2", "vola_s_w", "vola_s_w2"):
        arr = getattr(s, attr)
        assert arr.shape == (n,), f"{attr}.shape expected ({n},), got {arr.shape}"
    assert s.vola_count.shape == (n,)
    assert np.issubdtype(s.vola_count.dtype, np.integer)


@pytest.mark.parametrize("n", [1, 3, 10])
def test_pct_accumulator_shapes(n: int) -> None:
    """pct_s_* and pct_count must have shape (N,)."""
    s = _make_state(n)
    for attr in ("pct_s_x", "pct_s_x2", "pct_s_w", "pct_s_w2"):
        arr = getattr(s, attr)
        assert arr.shape == (n,), f"{attr}.shape expected ({n},), got {arr.shape}"
    assert s.pct_count.shape == (n,)
    assert np.issubdtype(s.pct_count.dtype, np.integer)


@pytest.mark.parametrize("n", [1, 3, 10])
def test_scalar_and_vector_fields(n: int) -> None:
    """Scalar and per-asset vector fields must have correct types/shapes."""
    s = _make_state(n)
    assert s.prev_price.shape == (n,)
    assert s.prev_cash_pos.shape == (n,)
    assert isinstance(s.step_count, int)
    assert s.step_count == 0


# ─── Mutability ───────────────────────────────────────────────────────────────


def test_step_count_is_mutable() -> None:
    """_StreamState is a plain (non-frozen) dataclass; fields must be mutable."""
    s = _make_state(2)
    s.step_count = 42
    assert s.step_count == 42


def test_array_field_replacement() -> None:
    """Array fields can be replaced with new arrays of the same shape."""
    n = 4
    s = _make_state(n)
    new_zi = np.ones((1, n, n))
    s.corr_zi_x = new_zi
    np.testing.assert_array_equal(s.corr_zi_x, new_zi)


def test_not_frozen() -> None:
    """_StreamState must NOT be frozen — FrozenInstanceError must not be raised."""
    s = _make_state(2)
    # If the dataclass were frozen this would raise dataclasses.FrozenInstanceError.
    try:
        s.step_count = 99
    except dataclasses.FrozenInstanceError:
        pytest.fail("_StreamState must be a mutable (non-frozen) dataclass")


# ─── Privacy ─────────────────────────────────────────────────────────────────


def test_not_exported_from_basanos_math() -> None:
    """_StreamState must NOT appear in the basanos.math public namespace."""
    import basanos.math as bm

    assert not hasattr(bm, "_StreamState"), "_StreamState should be private and not exported from basanos.math"


# ─── dataclass introspection ─────────────────────────────────────────────────


def test_is_dataclass() -> None:
    """_StreamState must be a proper dataclass."""
    assert dataclasses.is_dataclass(_StreamState)


def test_field_count() -> None:
    """_StreamState must expose exactly the documented fields."""
    fields = {f.name for f in dataclasses.fields(_StreamState)}
    expected = {
        "corr_zi_x",
        "corr_zi_x2",
        "corr_zi_xy",
        "corr_zi_w",
        "corr_count",
        "vola_s_x",
        "vola_s_x2",
        "vola_s_w",
        "vola_s_w2",
        "vola_count",
        "pct_s_x",
        "pct_s_x2",
        "pct_s_w",
        "pct_s_w2",
        "pct_count",
        "prev_price",
        "prev_cash_pos",
        "step_count",
        "sw_ret_buf",
    }
    assert fields == expected


# ─── construction & field access ─────────────────────────────────────────────


def test_construction_stores_fields():
    """Fields must equal the values passed at construction."""
    cash = np.array([1000.0, -500.0])
    vola = np.array([0.012, 0.018])
    result = StepResult(date="2024-01-02", cash_position=cash, status="valid", vola=vola)
    assert result.date == "2024-01-02"
    np.testing.assert_array_equal(result.cash_position, cash)
    assert result.status == "valid"
    np.testing.assert_array_equal(result.vola, vola)


def test_construction_with_nan_positions():
    """Warmup rows use NaN cash_position and vola."""
    n = 3
    cash = np.full(n, np.nan)
    vola = np.full(n, np.nan)
    result = StepResult(date=None, cash_position=cash, status="warmup", vola=vola)
    assert result.status == "warmup"
    assert np.all(np.isnan(result.cash_position))
    assert np.all(np.isnan(result.vola))


# ─── frozen semantics ────────────────────────────────────────────────────────


def test_frozen_raises_on_field_assignment():
    """Assigning to any field of a frozen dataclass must raise FrozenInstanceError."""
    cash = np.array([0.0])
    vola = np.array([0.01])
    result = StepResult(date="2024-01-03", cash_position=cash, status="valid", vola=vola)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = "warmup"  # type: ignore[misc]


def test_step_result_is_dataclass():
    """StepResult must be recognised as a dataclass."""
    assert dataclasses.is_dataclass(StepResult)


# ─── status values ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("status", ["warmup", "zero_signal", "degenerate", "valid"])
def test_all_valid_status_values(status: str):
    """All four documented status values must be storable without error."""
    result = StepResult(
        date=None,
        cash_position=np.zeros(2),
        status=status,
        vola=np.zeros(2),
    )
    assert result.status == status


# ─── public export ───────────────────────────────────────────────────────────


def test_exported_from_basanos_math():
    """StepResult imported from basanos.math must be the same class as in _stream."""
    assert StepResult is StepResultDirect


# ─── fixtures ─────────────────────────────────────────────────────────────────


def _make_prices_mu(n_total: int = 80, n_assets: int = 3, seed: int = 42):
    """Return (prices, mu, cfg, assets) for integration tests."""
    rng = np.random.default_rng(seed)
    dates = pl.date_range(
        start=date(2024, 1, 1),
        end=date(2024, 1, 1) + timedelta(days=n_total - 1),
        interval="1d",
        eager=True,
    )
    asset_names = [chr(ord("A") + i) for i in range(n_assets)]
    price_data = {
        "date": dates,
        **{a: np.cumprod(1 + rng.normal(0.001, 0.02, n_total)) * (100.0 + i * 50) for i, a in enumerate(asset_names)},
    }
    mu_data = {
        "date": dates,
        **{a: rng.normal(0, 0.5, n_total) for a in asset_names},
    }
    cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)
    return pl.DataFrame(price_data), pl.DataFrame(mu_data), cfg, asset_names


# ─── BasanosStream exports ────────────────────────────────────────────────────


def test_basanos_stream_exported():
    """BasanosStream must be importable from basanos.math."""
    assert BasanosStream is not None


def test_stream_state_importable_from_private_module():
    """_StreamState must be importable from the private basanos.math._stream module."""
    assert _StreamState is not None


def test_stream_state_is_dataclass():
    """_StreamState must be a dataclass."""
    assert dataclasses.is_dataclass(_StreamState)


# ─── from_warmup validation ───────────────────────────────────────────────────


def test_from_warmup_accepts_sliding_window():
    """from_warmup must succeed for SlidingWindowConfig and seed the rolling buffer."""
    prices, mu, _, assets = _make_prices_mu(n_total=80)
    cfg_sw = BasanosConfig(
        vola=5,
        corr=10,
        clip=3.0,
        shrink=0.5,
        aum=1e6,
        covariance_config=SlidingWindowConfig(window=20, n_factors=2),
    )
    stream = BasanosStream.from_warmup(prices.head(50), mu.head(50), cfg_sw)
    assert stream._state.sw_ret_buf is not None
    assert stream._state.sw_ret_buf.shape == (20, len(assets))


def test_from_warmup_rejects_missing_date_column():
    """from_warmup raises MissingDateColumnError when 'date' is absent."""
    prices, mu, cfg, assets = _make_prices_mu()
    prices_no_date = prices.select(assets)
    with pytest.raises(MissingDateColumnError):
        BasanosStream.from_warmup(prices_no_date.head(40), mu.head(40), cfg)


def test_from_warmup_returns_basanos_stream():
    """from_warmup must return a BasanosStream instance."""
    prices, mu, cfg, _ = _make_prices_mu()
    stream = BasanosStream.from_warmup(prices.head(50), mu.head(50), cfg)
    assert isinstance(stream, BasanosStream)


def test_from_warmup_assets_property():
    """Assets property must match the numeric columns of the warmup prices."""
    prices, mu, cfg, asset_names = _make_prices_mu()
    stream = BasanosStream.from_warmup(prices.head(50), mu.head(50), cfg)
    assert stream.assets == asset_names


# ─── acceptance criterion: step() matches BasanosEngine ───────────────────────


@pytest.mark.parametrize("warmup_len", [20, 40, 60])
def test_step_matches_basanos_engine(warmup_len: int):
    """step() must match BasanosEngine.cash_position[-1] within rtol=1e-8."""
    prices, mu, cfg, assets = _make_prices_mu(n_total=warmup_len + 1)
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])

    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
    expected = engine.cash_position.select(assets).to_numpy()[-1]

    np.testing.assert_allclose(result.cash_position, expected, rtol=1e-8, equal_nan=True)


@pytest.mark.parametrize("n_steps", [5, 10])
def test_multi_step_matches_basanos_engine(n_steps: int):
    """5+ consecutive step() calls must match BasanosEngine.cash_position within rtol=1e-8."""
    warmup_len = 60
    prices, mu, cfg, assets = _make_prices_mu(n_total=warmup_len + n_steps)
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)

    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
    expected = engine.cash_position.select(assets).to_numpy()

    for step_i in range(n_steps):
        row_idx = warmup_len + step_i
        result = stream.step(prices_np[row_idx], mu_np[row_idx], prices["date"][row_idx])
        np.testing.assert_allclose(
            result.cash_position,
            expected[row_idx],
            rtol=1e-8,
            equal_nan=True,
            err_msg=f"Mismatch at step {step_i + 1} (row {row_idx})",
        )


def test_step_returns_step_result():
    """step() must return a StepResult instance."""
    prices, mu, cfg, assets = _make_prices_mu()
    warmup_len = 50
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])
    assert isinstance(result, StepResult)


def test_step_cash_position_shape():
    """cash_position must have shape (N,) matching the number of assets."""
    prices, mu, cfg, assets = _make_prices_mu()
    warmup_len = 50
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])
    assert result.cash_position.shape == (len(assets),)


def test_step_vola_shape():
    """Vola must have shape (N,) matching the number of assets."""
    prices, mu, cfg, assets = _make_prices_mu()
    warmup_len = 50
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])
    assert result.vola.shape == (len(assets),)


def test_step_wrong_prices_shape_raises():
    """step() must raise ValueError with an informative message for wrong-length new_prices."""
    prices, mu, cfg, assets = _make_prices_mu()
    warmup_len = 50
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    bad_prices = prices_np[warmup_len, : len(assets) - 1]  # one element too few
    with pytest.raises(ValueError, match=r"new_prices must have shape"):
        stream.step(bad_prices, mu_np[warmup_len])


def test_step_wrong_mu_shape_raises():
    """step() must raise ValueError with an informative message for wrong-length new_mu."""
    prices, mu, cfg, assets = _make_prices_mu()
    warmup_len = 50
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    bad_mu = mu_np[warmup_len, : len(assets) - 1]  # one element too few
    with pytest.raises(ValueError, match=r"new_mu must have shape"):
        stream.step(prices_np[warmup_len], bad_mu)


def test_step_date_stored():
    """The date passed to step() must appear in StepResult.date."""
    prices, mu, cfg, assets = _make_prices_mu()
    warmup_len = 50
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()
    step_date = prices["date"][warmup_len]

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    result = stream.step(prices_np[warmup_len], mu_np[warmup_len], step_date)
    assert result.date == step_date


def test_step_dict_input_matches_array_input():
    """step() must produce identical results whether prices/mu are dicts or arrays."""
    prices, mu, cfg, assets = _make_prices_mu()
    warmup_len = 50
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    stream_arr = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    stream_dict = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)

    result_arr = stream_arr.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])
    p_dict = {a: float(prices_np[warmup_len, i]) for i, a in enumerate(assets)}
    m_dict = {a: float(mu_np[warmup_len, i]) for i, a in enumerate(assets)}
    result_dict = stream_dict.step(p_dict, m_dict, prices["date"][warmup_len])

    np.testing.assert_array_equal(result_arr.cash_position, result_dict.cash_position)


def test_step_status_is_valid_string():
    """step() status must be one of the four documented values."""
    prices, mu, cfg, assets = _make_prices_mu()
    warmup_len = 50
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])
    assert result.status in {"warmup", "zero_signal", "degenerate", "valid"}


def test_step_zero_signal_yields_zero_positions():
    """When all mu values are zero, cash_position must be all-zero (not NaN)."""
    prices, mu, cfg, assets = _make_prices_mu()
    warmup_len = 50
    prices_np = prices.select(assets).to_numpy()

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    zero_mu = dict.fromkeys(assets, 0.0)
    result = stream.step(prices_np[warmup_len], zero_mu, prices["date"][warmup_len])

    assert result.status == "zero_signal"
    np.testing.assert_array_equal(result.cash_position[np.isfinite(prices_np[warmup_len])], 0.0)


# ─── New required tests ───────────────────────────────────────────────────────


def test_stream_assets_match_engine():
    """stream.assets must equal BasanosEngine(...).assets for the same inputs."""
    prices, mu, cfg, _ = _make_prices_mu()
    warmup_len = 50
    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    engine = BasanosEngine(prices=prices.head(warmup_len), mu=mu.head(warmup_len), cfg=cfg)
    assert stream.assets == engine.assets


def test_stream_is_frozen():
    """Assigning any attribute on a BasanosStream instance must raise FrozenInstanceError."""
    prices, mu, cfg, _ = _make_prices_mu()
    stream = BasanosStream.from_warmup(prices.head(50), mu.head(50), cfg)
    with pytest.raises(dataclasses.FrozenInstanceError):
        stream.assets = ["X"]  # type: ignore[misc]


def test_stream_warmup_status_before_min_periods():
    """step() returns status='warmup' when the warmup batch is shorter than cfg.corr."""
    # Use deterministic oscillating prices to avoid MonotonicPricesError with few rows
    warmup_len = 5
    n_total = warmup_len + 1
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n_total - 1), interval="1d", eager=True)
    # Alternating prices guarantee non-monotonic (required by the engine)
    price_a = np.array([100.0, 102.0, 101.0, 103.0, 102.0, 104.0], dtype=float)
    price_b = np.array([200.0, 198.0, 202.0, 199.0, 203.0, 200.0], dtype=float)
    prices = pl.DataFrame({"date": dates, "A": price_a, "B": price_b})
    mu_vals = np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.1], dtype=float)
    mu = pl.DataFrame({"date": dates, "A": mu_vals, "B": mu_vals})
    cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)

    # warmup_len=5 < cfg.corr=10, so step_count (=5) < cfg.corr → warmup
    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    assets = stream.assets
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])

    assert result.status == "warmup"
    assert np.all(np.isnan(result.cash_position))
    assert np.all(np.isnan(result.vola))


def test_stream_warmup_skips_solve_state_unchanged():
    """Warmup step() must not update prev_cash_pos.

    The early-return guard skips the matrix reconstruction and solve block, so
    *state.prev_cash_pos* must remain unchanged (all-NaN from warmup or equal
    to its initial value) after every warmup step.
    """
    warmup_len = 5
    n_total = warmup_len + 5
    rng = np.random.default_rng(42)
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n_total - 1), interval="1d", eager=True)
    price_a = np.array([100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0, 104.0, 106.0], dtype=float)
    price_b = np.array([200.0, 198.0, 202.0, 199.0, 203.0, 200.0, 204.0, 201.0, 205.0, 202.0], dtype=float)
    prices = pl.DataFrame({"date": dates, "A": price_a, "B": price_b})
    mu_vals = rng.normal(0, 0.5, n_total)
    mu = pl.DataFrame({"date": dates, "A": mu_vals, "B": mu_vals})
    cfg = BasanosConfig(vola=5, corr=20, clip=3.0, shrink=0.5, aum=1e6)

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    assert stream._state.step_count < cfg.corr, "fixture: stream must start in warmup"

    assets = stream.assets
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    # Record state before any streaming step
    initial_prev_cash = stream._state.prev_cash_pos.copy()

    for i in range(n_total - warmup_len):
        assert stream._state.step_count < cfg.corr, "all steps in this loop must be warmup"
        result = stream.step(prices_np[warmup_len + i], mu_np[warmup_len + i])

        assert result.status == "warmup"
        # The early-return path must not touch prev_cash_pos
        assert np.all(np.isnan(stream._state.prev_cash_pos)) or np.array_equal(
            stream._state.prev_cash_pos, initial_prev_cash
        ), "prev_cash_pos changed during warmup; solve block may not be short-circuited"


def test_stream_warmup_step_skips_shrink2id():
    """Warmup step() must short-circuit before calling shrink2id.

    Patch ``shrink2id`` in ``basanos.math._stream`` and assert it is **not**
    called during warmup steps (step_count < cfg.corr) but **is** called
    during post-warmup steps (step_count >= cfg.corr).  This verifies the
    early-return guard at the source level without any wallclock dependency.
    """
    n_assets = 5
    n_steps = 3

    cfg = BasanosConfig(vola=5, corr=60, clip=3.0, shrink=0.5, aum=1e6)
    short_warmup = 10  # step_count = 10 < cfg.corr = 60  →  all steps are warmup
    long_warmup = 60  # step_count = 60 = cfg.corr  →  all steps are real solves
    n_total = long_warmup + n_steps
    rng = np.random.default_rng(7)
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n_total - 1), interval="1d", eager=True)
    asset_names = [chr(ord("A") + i) for i in range(n_assets)]

    price_data: dict[str, object] = {"date": dates}
    mu_data: dict[str, object] = {"date": dates}
    for a in asset_names:
        base = 100.0 + rng.uniform(0, 100)
        t = np.arange(n_total, dtype=float)
        price_data[a] = np.abs(base + np.cumsum(rng.normal(0, 0.5, n_total)) + 5 * np.sin(t * 0.3)) + 1.0
        mu_data[a] = rng.normal(0, 0.5, n_total)

    prices = pl.DataFrame(price_data)
    mu = pl.DataFrame(mu_data)
    prices_np = prices.select(asset_names).to_numpy()
    mu_np = mu.select(asset_names).to_numpy()

    warmup_stream = BasanosStream.from_warmup(prices.head(short_warmup), mu.head(short_warmup), cfg)
    assert warmup_stream._state.step_count < cfg.corr, "fixture: stream must start in warmup"

    full_stream = BasanosStream.from_warmup(prices.head(long_warmup), mu.head(long_warmup), cfg)
    assert full_stream._state.step_count >= cfg.corr, "fixture: stream must be past warmup"

    # shrink2id must NOT be called during warmup steps
    with patch("basanos.math._stream.shrink2id", wraps=None) as mock_shrink:
        for i in range(n_steps):
            warmup_stream.step(prices_np[short_warmup + i], mu_np[short_warmup + i])
        assert mock_shrink.call_count == 0, (
            f"shrink2id was called {mock_shrink.call_count} time(s) during warmup; "
            "the early-return guard may be missing or broken"
        )

    # shrink2id must be called for every post-warmup step
    import basanos.math._stream as _stream_mod

    real_shrink2id = _stream_mod.shrink2id
    with patch("basanos.math._stream.shrink2id", wraps=real_shrink2id) as mock_shrink:
        for i in range(n_steps):
            full_stream.step(prices_np[long_warmup + i], mu_np[long_warmup + i])
        assert mock_shrink.call_count == n_steps, (
            f"shrink2id called {mock_shrink.call_count} time(s) but expected {n_steps} "
            "for post-warmup steps; the solve block may not be reached"
        )


@pytest.mark.slow
def test_stream_warmup_step_faster_than_full_solve():
    """Warmup step() must short-circuit before the O(N**2) matrix solve.

    Compare the per-step latency of a warmup stream (step_count < cfg.corr)
    against a fully-warmed-up stream (step_count >= cfg.corr) using identical
    inputs.  Warmup steps must be at least 1.5x faster because the
    correlation-matrix reconstruction and Cholesky solve are skipped.
    The 1.5x threshold provides tolerance for system-load variation in CI.

    Note: this test is marked ``@pytest.mark.slow`` and skipped in normal CI
    runs because wallclock measurements are susceptible to scheduling noise on
    shared runners.  The behavioural guard is covered by
    ``test_stream_warmup_step_skips_shrink2id``.
    """
    n_assets = 20  # large enough that O(N²) work is measurable
    n_steps = 30

    # corr=60: short batch (10 rows) stays in warmup; long batch (60 rows) exits it
    cfg = BasanosConfig(vola=5, corr=60, clip=3.0, shrink=0.5, aum=1e6)
    short_warmup = 10
    long_warmup = 60
    n_total = long_warmup + n_steps
    rng = np.random.default_rng(7)
    start = date(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n_total - 1), interval="1d", eager=True)
    asset_names = [chr(ord("A") + i) for i in range(n_assets)]

    price_data: dict[str, object] = {"date": dates}
    mu_data: dict[str, object] = {"date": dates}
    for a in asset_names:
        base = 100.0 + rng.uniform(0, 100)
        t = np.arange(n_total, dtype=float)
        price_data[a] = np.abs(base + np.cumsum(rng.normal(0, 0.5, n_total)) + 5 * np.sin(t * 0.3)) + 1.0
        mu_data[a] = rng.normal(0, 0.5, n_total)

    prices = pl.DataFrame(price_data)
    mu = pl.DataFrame(mu_data)
    prices_np = prices.select(asset_names).to_numpy()
    mu_np = mu.select(asset_names).to_numpy()

    # Warmup stream: step_count = 10 < cfg.corr = 60  →  all steps are warmup
    warmup_stream = BasanosStream.from_warmup(prices.head(short_warmup), mu.head(short_warmup), cfg)
    assert warmup_stream._state.step_count < cfg.corr

    # Post-warmup stream: step_count = 60 = cfg.corr  →  all steps are real solves
    full_stream = BasanosStream.from_warmup(prices.head(long_warmup), mu.head(long_warmup), cfg)
    assert full_stream._state.step_count >= cfg.corr

    t0 = time.perf_counter()
    for i in range(n_steps):
        warmup_stream.step(prices_np[short_warmup + i], mu_np[short_warmup + i])
    warmup_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    for i in range(n_steps):
        full_stream.step(prices_np[long_warmup + i], mu_np[long_warmup + i])
    full_elapsed = time.perf_counter() - t1

    warmup_per_step = warmup_elapsed / n_steps
    full_per_step = full_elapsed / n_steps

    # Warmup per-step latency must be at least 1.5x faster than a full solve.
    # The 1.5x factor provides tolerance for CI system-load variation while
    # still reliably catching regressions where the early-return is absent.
    assert warmup_per_step * 1.5 < full_per_step, (
        f"Warmup step ({warmup_per_step * 1e6:.1f} us) not at least 1.5x faster than "
        f"full-solve step ({full_per_step * 1e6:.1f} us); "
        "the early-return guard may be missing or broken"
    )


def test_stream_degenerate_status_all_nan_prices():
    """step() returns status='degenerate' when the new price row is all-NaN."""
    prices, mu, cfg, assets = _make_prices_mu()
    warmup_len = 50
    mu_np = mu.select(assets).to_numpy()

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    all_nan_prices = np.full(len(assets), np.nan)
    result = stream.step(all_nan_prices, mu_np[warmup_len], prices["date"][warmup_len])

    assert result.status == "degenerate"


# ─── Hypothesis property tests ────────────────────────────────────────────────


@given(
    n_assets=st.integers(min_value=1, max_value=3),
    n_rows=st.integers(min_value=5, max_value=40),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=40, deadline=3000)  # 3 s per example; stream warmup is O(N²)
def test_hypothesis_stream_properties(n_assets: int, n_rows: int, seed: int) -> None:
    """For arbitrary small inputs: status ∈ valid set and no ±inf in cash_position."""
    from basanos.exceptions import MonotonicPricesError

    rng = np.random.default_rng(seed)
    start = date(2020, 1, 1)
    n_total = n_rows + 1  # warmup of n_rows, then one step
    dates = pl.date_range(
        start=start,
        end=start + timedelta(days=n_total - 1),
        interval="1d",
        eager=True,
    )
    asset_names = [chr(ord("A") + i) for i in range(n_assets)]
    price_data: dict[str, object] = {"date": dates}
    mu_data: dict[str, object] = {"date": dates}
    for a in asset_names:
        # Use sinusoidal perturbations on top of random walk to prevent monotonic series
        base = 100.0 + rng.uniform(0, 100)
        noise = rng.normal(0, 0.01, n_total)
        osc = 0.01 * np.sin(np.linspace(0, 4 * np.pi, n_total))
        price_data[a] = base * np.cumprod(1 + noise + osc)
        mu_data[a] = rng.normal(0, 0.5, n_total)

    prices = pl.DataFrame(price_data)
    mu_df = pl.DataFrame(mu_data)
    cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)

    try:
        stream = BasanosStream.from_warmup(prices.head(n_rows), mu_df.head(n_rows), cfg)
    except MonotonicPricesError:
        # Random price series can still be monotonic with few rows; skip this example
        return

    prices_np = prices.select(asset_names).to_numpy()
    mu_np = mu_df.select(asset_names).to_numpy()

    result = stream.step(prices_np[n_rows], mu_np[n_rows], prices["date"][n_rows])

    valid_statuses = {"warmup", "zero_signal", "degenerate", "valid"}
    assert result.status in valid_statuses, f"Unexpected status: {result.status!r}"

    cp = result.cash_position
    assert not np.any(np.isposinf(cp)), f"+inf in cash_position: {cp}"
    assert not np.any(np.isneginf(cp)), f"-inf in cash_position: {cp}"


# ─── warmup_state() tests ─────────────────────────────────────────────────────


def test_warmup_state_returns_warmup_state():
    """warmup_state() must return a WarmupState instance."""
    prices, mu, cfg, _ = _make_prices_mu()
    engine = BasanosEngine(prices=prices.head(50), mu=mu.head(50), cfg=cfg)
    ws = engine.warmup_state()
    assert isinstance(ws, WarmupState)


def test_warmup_state_prev_cash_pos_shape():
    """prev_cash_pos must have shape (n_assets,)."""
    prices, mu, cfg, assets = _make_prices_mu()
    engine = BasanosEngine(prices=prices.head(50), mu=mu.head(50), cfg=cfg)
    ws = engine.warmup_state()
    assert ws.prev_cash_pos.shape == (len(assets),)


def test_warmup_state_is_frozen():
    """WarmupState must be frozen (immutable)."""
    prices, mu, cfg, _ = _make_prices_mu()
    engine = BasanosEngine(prices=prices.head(50), mu=mu.head(50), cfg=cfg)
    ws = engine.warmup_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ws.prev_cash_pos = np.zeros(2)  # type: ignore[misc]


def test_warmup_state_matches_from_warmup_state():
    """prev_cash_pos from warmup_state() must match the internal stream state."""
    prices, mu, cfg, _assets = _make_prices_mu()
    n = 50
    engine = BasanosEngine(prices=prices.head(n), mu=mu.head(n), cfg=cfg)
    ws = engine.warmup_state()

    stream = BasanosStream.from_warmup(prices.head(n), mu.head(n), cfg)
    np.testing.assert_array_equal(ws.prev_cash_pos, stream._state.prev_cash_pos)


def test_warmup_state_single_row():
    """warmup_state() must not raise for a single-row batch."""
    start = date(2024, 1, 1)
    dates = pl.date_range(start=start, end=start, interval="1d", eager=True)
    prices = pl.DataFrame({"date": dates, "A": [100.0], "B": [200.0]})
    mu = pl.DataFrame({"date": dates, "A": [0.1], "B": [-0.1]})
    cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
    ws = engine.warmup_state()
    assert isinstance(ws, WarmupState)
    assert ws.prev_cash_pos.shape == (2,)


# ─── Isolation tests: from_warmup state extraction ────────────────────────────
#
# These tests verify each piece of state extracted by from_warmup independently
# of the downstream solve step.  A bug in state extraction that happened to
# cancel out in the solve would be invisible to test_step_matches_basanos_engine
# but would be caught here.


def test_corr_iir_state_is_beta_times_last_filter_output():
    """corr_zi_x[0] must equal beta_corr * s_x[-1] from scipy lfilter on the warmup data.

    For a first-order IIR filter y[t] = beta*y[t-1] + x[t], the final filter
    memory (zf) satisfies zf[0] = beta * y[T-1].  from_warmup must store that
    memory in corr_zi_x so that the next step() call correctly continues the
    same recurrence.  This test reconstructs s_x[-1] from the same IIR inputs
    that _ewm_corr_numpy uses and verifies the stored memory matches, without
    exercising the downstream solve path.
    """
    from scipy.signal import lfilter as scipy_lfilter

    prices, mu, cfg, assets = _make_prices_mu(n_total=60)
    warmup_prices = prices.head(50)
    warmup_mu = mu.head(50)

    stream = BasanosStream.from_warmup(warmup_prices, warmup_mu, cfg)

    # Independently reconstruct the same IIR inputs that _ewm_corr_numpy uses
    engine = BasanosEngine(prices=warmup_prices, mu=warmup_mu, cfg=cfg)
    ret_adj_np = engine.ret_adj.select(assets).to_numpy()  # (T, N)

    beta_corr = cfg.corr / (1.0 + cfg.corr)
    fin = np.isfinite(ret_adj_np)
    xt_f = np.where(fin, ret_adj_np, 0.0)
    joint_fin = fin[:, :, np.newaxis] & fin[:, np.newaxis, :]

    v_x = xt_f[:, :, np.newaxis] * joint_fin
    v_x2 = (xt_f**2)[:, :, np.newaxis] * joint_fin
    v_xy = xt_f[:, :, np.newaxis] * xt_f[:, np.newaxis, :]
    v_w = joint_fin.astype(np.float64)

    filt_a = np.array([1.0, -beta_corr])
    # Run lfilter without zi (same as _ewm_corr_numpy) to get the last output.
    # The IIR memory after the last step is: zf[0] = beta * y[-1].
    s_x = scipy_lfilter([1.0], filt_a, v_x, axis=0)
    s_x2 = scipy_lfilter([1.0], filt_a, v_x2, axis=0)
    s_xy = scipy_lfilter([1.0], filt_a, v_xy, axis=0)
    s_w = scipy_lfilter([1.0], filt_a, v_w, axis=0)

    n = len(assets)
    expected_zi_x = np.zeros((1, n, n))
    expected_zi_x2 = np.zeros((1, n, n))
    expected_zi_xy = np.zeros((1, n, n))
    expected_zi_w = np.zeros((1, n, n))
    expected_zi_x[0] = beta_corr * s_x[-1]
    expected_zi_x2[0] = beta_corr * s_x2[-1]
    expected_zi_xy[0] = beta_corr * s_xy[-1]
    expected_zi_w[0] = beta_corr * s_w[-1]

    np.testing.assert_allclose(stream._state.corr_zi_x, expected_zi_x, rtol=1e-12)
    np.testing.assert_allclose(stream._state.corr_zi_x2, expected_zi_x2, rtol=1e-12)
    np.testing.assert_allclose(stream._state.corr_zi_xy, expected_zi_xy, rtol=1e-12)
    np.testing.assert_allclose(stream._state.corr_zi_w, expected_zi_w, rtol=1e-12)


def test_corr_count_matches_cumulative_joint_finite():
    """corr_count must equal the cumulative sum of joint-finite observations.

    Each entry corr_count[i, j] must equal the number of warmup rows at which
    both asset i and asset j had a finite vol-adjusted return — the same
    quantity that _ewm_corr_numpy accumulates in its count tensor.
    """
    prices, mu, cfg, assets = _make_prices_mu(n_total=60)
    warmup_prices = prices.head(50)
    warmup_mu = mu.head(50)

    stream = BasanosStream.from_warmup(warmup_prices, warmup_mu, cfg)

    engine = BasanosEngine(prices=warmup_prices, mu=warmup_mu, cfg=cfg)
    ret_adj_np = engine.ret_adj.select(assets).to_numpy()  # (T, N)

    fin = np.isfinite(ret_adj_np)  # (T, N)
    joint_fin = fin[:, :, np.newaxis] & fin[:, np.newaxis, :]  # (T, N, N)
    expected_count = joint_fin.astype(np.int64).sum(axis=0)  # (N, N)

    np.testing.assert_array_equal(stream._state.corr_count, expected_count)


def test_pct_accumulators_match_polars_ewm_std():
    """_ewm_std_from_state(pct_s_*) for the warmup last row must match engine.vola.

    engine.vola uses polars.Expr.ewm_std(com=cfg.vola-1, adjust=True).
    This test verifies that the pct_s_* accumulators extracted by from_warmup
    encode exactly the same information, independently of the solve path.
    A discrepancy here means from_warmup's volatility state is stale relative
    to the Polars accumulator semantics.
    """
    prices, mu, cfg, assets = _make_prices_mu(n_total=60)
    warmup_prices = prices.head(50)
    warmup_mu = mu.head(50)

    stream = BasanosStream.from_warmup(warmup_prices, warmup_mu, cfg)
    state = stream._state

    # Compute volatility from the extracted accumulators
    stream_vola = _ewm_std_from_state(
        state.pct_s_x,
        state.pct_s_x2,
        state.pct_s_w,
        state.pct_s_w2,
        state.pct_count,
        min_samples=cfg.vola,
    )

    # Expected: Polars batch ewm_std on the same warmup data (last row)
    engine = BasanosEngine(prices=warmup_prices, mu=warmup_mu, cfg=cfg)
    expected_vola = engine.vola.select(assets).to_numpy()[-1]  # last warmup row

    np.testing.assert_allclose(stream_vola, expected_vola, rtol=1e-8, equal_nan=True)


def test_vola_log_accumulators_match_batch_ewm_std():
    """_ewm_std_from_state(vola_s_*) must match an independent batch EWM std of log-returns.

    vola_s_* tracks the EWMA of log-returns for vol-adjustment (min_samples=1).
    This test verifies the accumulators agree with an independent lfilter
    computation on the same log-return series, without going through the solve.
    """
    from scipy.signal import lfilter as scipy_lfilter

    prices, mu, cfg, assets = _make_prices_mu(n_total=60)
    warmup_prices = prices.head(50)
    warmup_mu = mu.head(50)

    stream = BasanosStream.from_warmup(warmup_prices, warmup_mu, cfg)
    state = stream._state

    # Reconstruct log-returns independently
    prices_np = warmup_prices.select(assets).to_numpy()
    n_rows = prices_np.shape[0]
    log_ret = np.full_like(prices_np, np.nan)
    if n_rows > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret[1:] = np.log(prices_np[1:] / prices_np[:-1])

    beta_vola = (cfg.vola - 1) / cfg.vola
    beta_vola_sq = beta_vola**2

    fin_log = np.isfinite(log_ret).astype(np.float64)
    log_ret_z = np.where(fin_log.astype(bool), log_ret, 0.0)
    filt_a = np.array([1.0, -beta_vola])
    filt_a2 = np.array([1.0, -beta_vola_sq])

    expected_s_x = scipy_lfilter([1.0], filt_a, log_ret_z, axis=0)[-1]
    expected_s_x2 = scipy_lfilter([1.0], filt_a, log_ret_z**2, axis=0)[-1]
    expected_s_w = scipy_lfilter([1.0], filt_a, fin_log, axis=0)[-1]
    expected_s_w2 = scipy_lfilter([1.0], filt_a2, fin_log, axis=0)[-1]
    expected_count = fin_log.sum(axis=0).astype(int)

    np.testing.assert_allclose(state.vola_s_x, expected_s_x, rtol=1e-12)
    np.testing.assert_allclose(state.vola_s_x2, expected_s_x2, rtol=1e-12)
    np.testing.assert_allclose(state.vola_s_w, expected_s_w, rtol=1e-12)
    np.testing.assert_allclose(state.vola_s_w2, expected_s_w2, rtol=1e-12)
    np.testing.assert_array_equal(state.vola_count, expected_count)


# ─── Tests for the _ewm_vol_accumulators_from_batch helper ────────────────────


def test_ewm_vol_accumulators_scalar_recurrence_identity():
    """Batch helper must match the scalar step-by-step recurrence in step().

    BasanosStream.step() advances the accumulators one row at a time using the
    scalar recurrence s_x[t] = beta*s_x[t-1] + x[t].  _ewm_vol_accumulators_from_batch
    achieves the same result in one vectorised lfilter call.  If these two paths
    diverge, from_warmup's initial state would be inconsistent with the incremental
    updates in step(), causing a discontinuity at the warmup boundary.
    """
    rng = np.random.default_rng(99)
    t_len = 40
    n_assets = 3
    cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)
    beta = (cfg.vola - 1) / cfg.vola
    beta_sq = beta**2

    # Random returns with a few NaNs
    returns = rng.normal(0, 0.01, (t_len, n_assets))
    returns[0] = np.nan  # typical leading NaN from pct_change / log-diff
    returns[5, 1] = np.nan  # sporadic NaN

    # Batch helper (lfilter path)
    s_x_b, s_x2_b, s_w_b, s_w2_b, count_b = _ewm_vol_accumulators_from_batch(returns, beta, beta_sq)

    # Scalar step-by-step recurrence (mirrors BasanosStream.step)
    s_x_s = np.zeros(n_assets)
    s_x2_s = np.zeros(n_assets)
    s_w_s = np.zeros(n_assets)
    s_w2_s = np.zeros(n_assets)
    count_s = np.zeros(n_assets, dtype=int)
    for row in returns:
        fin = np.isfinite(row)
        s_x_s = beta * s_x_s + np.where(fin, row, 0.0)
        s_x2_s = beta * s_x2_s + np.where(fin, row**2, 0.0)
        s_w_s = beta * s_w_s + fin.astype(float)
        s_w2_s = beta_sq * s_w2_s + fin.astype(float)
        count_s = count_s + fin.astype(int)

    np.testing.assert_allclose(s_x_b, s_x_s, rtol=1e-12)
    np.testing.assert_allclose(s_x2_b, s_x2_s, rtol=1e-12)
    np.testing.assert_allclose(s_w_b, s_w_s, rtol=1e-12)
    np.testing.assert_allclose(s_w2_b, s_w2_s, rtol=1e-12)
    np.testing.assert_array_equal(count_b, count_s)


def test_ewm_vol_accumulators_all_nan_returns_zero_accumulators():
    """When every return is NaN, all accumulators must be zero and count zero."""
    n_assets = 4
    t_len = 20
    returns = np.full((t_len, n_assets), np.nan)
    beta = 0.8
    beta_sq = beta**2

    s_x, s_x2, s_w, s_w2, count = _ewm_vol_accumulators_from_batch(returns, beta, beta_sq)

    np.testing.assert_array_equal(s_x, np.zeros(n_assets))
    np.testing.assert_array_equal(s_x2, np.zeros(n_assets))
    np.testing.assert_array_equal(s_w, np.zeros(n_assets))
    np.testing.assert_array_equal(s_w2, np.zeros(n_assets))
    np.testing.assert_array_equal(count, np.zeros(n_assets, dtype=int))


def test_ewm_vol_accumulators_output_shapes():
    """Output arrays must have shape (N,) and count must be integer dtype."""
    n_assets = 5
    t_len = 15
    returns = np.random.default_rng(7).normal(0, 0.01, (t_len, n_assets))
    beta = 0.75
    beta_sq = beta**2

    s_x, s_x2, s_w, s_w2, count = _ewm_vol_accumulators_from_batch(returns, beta, beta_sq)

    for arr, name in [(s_x, "s_x"), (s_x2, "s_x2"), (s_w, "s_w"), (s_w2, "s_w2")]:
        assert arr.shape == (n_assets,), f"{name}.shape expected ({n_assets},), got {arr.shape}"
    assert count.shape == (n_assets,)
    assert np.issubdtype(count.dtype, np.integer)


# ─── corr_iir_state tests ──────────────────────────────────────────────────────


def test_warmup_state_has_corr_iir_state_for_ewma_shrink():
    """corr_iir_state must be non-None for EwmaShrinkConfig."""
    prices, mu, cfg, _ = _make_prices_mu()
    engine = BasanosEngine(prices=prices.head(50), mu=mu.head(50), cfg=cfg)
    ws = engine.warmup_state()
    assert ws.corr_iir_state is not None


def test_warmup_state_corr_iir_state_is_none_for_sliding_window():
    """corr_iir_state must be None for SlidingWindowConfig."""
    prices, mu, _, _assets = _make_prices_mu(n_total=80)
    cfg = BasanosConfig(
        vola=5,
        corr=10,
        clip=3.0,
        shrink=0.5,
        aum=1e6,
        covariance_config=SlidingWindowConfig(window=20, n_factors=2),
    )
    engine = BasanosEngine(prices=prices.head(60), mu=mu.head(60), cfg=cfg)
    ws = engine.warmup_state()
    assert ws.corr_iir_state is None


def test_warmup_state_corr_iir_state_shapes():
    """corr_iir_state fields must have documented shapes."""
    prices, mu, cfg, assets = _make_prices_mu()
    n_assets = len(assets)
    engine = BasanosEngine(prices=prices.head(50), mu=mu.head(50), cfg=cfg)
    ws = engine.warmup_state()
    iir = ws.corr_iir_state
    assert iir is not None
    assert iir.corr_zi_x.shape == (1, n_assets, n_assets)
    assert iir.corr_zi_x2.shape == (1, n_assets, n_assets)
    assert iir.corr_zi_xy.shape == (1, n_assets, n_assets)
    assert iir.corr_zi_w.shape == (1, n_assets, n_assets)
    assert iir.count.shape == (n_assets, n_assets)
    assert iir.count.dtype == np.int64


def test_warmup_state_corr_iir_state_matches_stream_state():
    """IIR state from warmup_state() must match the IIR state seeded into _StreamState by from_warmup()."""
    prices, mu, cfg, _assets = _make_prices_mu()
    n = 50
    engine = BasanosEngine(prices=prices.head(n), mu=mu.head(n), cfg=cfg)
    ws = engine.warmup_state()
    iir = ws.corr_iir_state
    assert iir is not None

    stream = BasanosStream.from_warmup(prices.head(n), mu.head(n), cfg)
    np.testing.assert_allclose(iir.corr_zi_x, stream._state.corr_zi_x, rtol=1e-12)
    np.testing.assert_allclose(iir.corr_zi_x2, stream._state.corr_zi_x2, rtol=1e-12)
    np.testing.assert_allclose(iir.corr_zi_xy, stream._state.corr_zi_xy, rtol=1e-12)
    np.testing.assert_allclose(iir.corr_zi_w, stream._state.corr_zi_w, rtol=1e-12)
    np.testing.assert_array_equal(iir.count, stream._state.corr_count)


def test_warmup_state_ewma_zero_signal_row():
    """warmup_state() must not raise when a row has all-zero mu (zero_signal path)."""
    prices, mu, cfg, assets = _make_prices_mu()
    n = 50
    # Zero out the last row of mu to trigger the zero_signal branch
    mu_np = mu.head(n).select(assets).to_numpy().copy()
    mu_np[-1, :] = 0.0
    dates_col = mu.head(n)["date"]
    mu_zeroed = pl.DataFrame({"date": dates_col, **{a: mu_np[:, i] for i, a in enumerate(assets)}})
    engine = BasanosEngine(prices=prices.head(n), mu=mu_zeroed, cfg=cfg)
    ws = engine.warmup_state()
    assert isinstance(ws, WarmupState)
    # Last row had zero_signal; prev_cash_pos must be all-zero (not NaN)
    assert np.all(ws.prev_cash_pos == 0.0)


def test_warmup_state_ewma_all_nan_prices_row():
    """warmup_state() must not raise when a row has all-NaN prices (mask.any() == False path)."""
    from datetime import date as dt

    n = 50
    rng = np.random.default_rng(99)
    start = dt(2020, 1, 1)
    dates = pl.date_range(start=start, end=start + timedelta(days=n - 1), interval="1d", eager=True)

    # Build alternating non-monotonic prices for two assets
    price_a = np.cumprod(1 + np.where(np.arange(n) % 2 == 0, 0.01, -0.005)) * 100.0
    price_b = np.cumprod(1 + np.where(np.arange(n) % 2 == 0, -0.005, 0.01)) * 150.0
    mu_vals = rng.normal(0, 0.5, n)

    # Set one interior row to null for both assets (null_pct = 2/100 = 2% < 50%)
    price_a_list = price_a.tolist()
    price_b_list = price_b.tolist()
    price_a_list[20] = None
    price_b_list[20] = None

    prices = pl.DataFrame({"date": dates, "A": price_a_list, "B": price_b_list})
    mu = pl.DataFrame({"date": dates, "A": mu_vals, "B": mu_vals})
    cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)
    engine = BasanosEngine(prices=prices, mu=mu, cfg=cfg)
    ws = engine.warmup_state()
    assert isinstance(ws, WarmupState)
    assert ws.corr_iir_state is not None


def test_warmup_state_ewma_singular_inv_a_norm():
    """warmup_state() must not raise when inv_a_norm raises SingularMatrixError (sets denom=nan)."""
    from unittest.mock import patch

    from basanos.exceptions import SingularMatrixError

    prices, mu, cfg, _ = _make_prices_mu()
    n = 50
    engine = BasanosEngine(prices=prices.head(n), mu=mu.head(n), cfg=cfg)

    with (
        patch.object(np.linalg, "solve", side_effect=np.linalg.LinAlgError("singular")),
        patch("basanos.math._engine_solve.inv_a_norm", side_effect=SingularMatrixError("singular")),
    ):
        ws = engine.warmup_state()

    assert isinstance(ws, WarmupState)
    # All positions were zeroed due to degenerate denom
    assert np.all(ws.prev_cash_pos == 0.0)


def test_warmup_state_ewma_degenerate_denom():
    """warmup_state() must not raise when inv_a_norm returns a non-finite denom."""
    from unittest.mock import patch

    prices, mu, cfg, _ = _make_prices_mu()
    n = 50
    engine = BasanosEngine(prices=prices.head(n), mu=mu.head(n), cfg=cfg)

    with (
        patch.object(np.linalg, "solve", side_effect=np.linalg.LinAlgError("singular")),
        patch("basanos.math._engine_solve.inv_a_norm", return_value=float("nan")),
    ):
        ws = engine.warmup_state()

    assert isinstance(ws, WarmupState)
    assert np.all(ws.prev_cash_pos == 0.0)


def test_warmup_state_ewma_singular_solve():
    """warmup_state() must not raise when solve raises SingularMatrixError after inv_a_norm succeeds."""
    from unittest.mock import patch

    from basanos.exceptions import SingularMatrixError

    prices, mu, cfg, _ = _make_prices_mu()
    n = 50
    engine = BasanosEngine(prices=prices.head(n), mu=mu.head(n), cfg=cfg)

    # Force the batched path to fall back to _compute_position, then make solve fail there.
    with (
        patch.object(np.linalg, "solve", side_effect=np.linalg.LinAlgError("singular")),
        patch("basanos.math._engine_solve.solve", side_effect=SingularMatrixError("singular")),
    ):
        ws = engine.warmup_state()

    assert isinstance(ws, WarmupState)
    assert np.all(ws.prev_cash_pos == 0.0)


# ─── _ewm_std_from_state edge cases ──────────────────────────────────────────


def test_ewm_std_from_state_all_counts_below_min_samples_returns_all_nan():
    """_ewm_std_from_state must return all-NaN immediately when no asset meets min_samples."""
    n = 4
    s_x = np.ones(n)
    s_x2 = np.ones(n)
    s_w = np.ones(n)
    s_w2 = np.ones(n)
    count = np.zeros(n, dtype=int)  # all below any min_samples >= 1

    result = _ewm_std_from_state(s_x, s_x2, s_w, s_w2, count, min_samples=1)

    assert result.shape == (n,)
    assert np.all(np.isnan(result))


# ─── step() error-handling branches ──────────────────────────────────────────


def test_step_singular_inv_a_norm_yields_degenerate():
    """step() must return status='degenerate' when inv_a_norm raises SingularMatrixError."""
    from basanos.exceptions import SingularMatrixError

    prices, mu, cfg, assets = _make_prices_mu()
    warmup_len = 50
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)

    with patch("basanos.math._stream.inv_a_norm", side_effect=SingularMatrixError("singular")):
        result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])

    assert result.status == "degenerate"
    assert np.all(result.cash_position == 0.0)


def test_step_singular_solve_yields_degenerate():
    """step() must return status='degenerate' when solve raises SingularMatrixError."""
    from basanos.exceptions import SingularMatrixError

    prices, mu, cfg, assets = _make_prices_mu()
    warmup_len = 50
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)

    with patch("basanos.math._stream.solve", side_effect=SingularMatrixError("singular")):
        result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])

    assert result.status == "degenerate"
    assert np.all(result.cash_position == 0.0)


# ─── save / load round-trip ───────────────────────────────────────────────────


def test_save_load_roundtrip_produces_identical_step_output():
    """A stream restored from disk must produce bit-for-bit identical step() output."""
    import pathlib
    import tempfile

    prices, mu, cfg, assets = _make_prices_mu()
    warmup_len = 50
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    stream_orig = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)

    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "stream.npz"
        stream_orig.save(p)
        stream_loaded = BasanosStream.load(p)

    step_date = prices["date"][warmup_len]
    result_orig = stream_orig.step(prices_np[warmup_len], mu_np[warmup_len], step_date)
    result_loaded = stream_loaded.step(prices_np[warmup_len], mu_np[warmup_len], step_date)

    np.testing.assert_array_equal(result_orig.cash_position, result_loaded.cash_position)
    np.testing.assert_array_equal(result_orig.vola, result_loaded.vola)
    assert result_orig.status == result_loaded.status


def test_save_load_preserves_assets_and_cfg():
    """Restored stream must have the same assets list and config as the original."""
    import pathlib
    import tempfile

    prices, mu, cfg, _assets = _make_prices_mu()

    stream = BasanosStream.from_warmup(prices.head(50), mu.head(50), cfg)

    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "stream.npz"
        stream.save(p)
        restored = BasanosStream.load(p)

    assert restored.assets == stream.assets
    assert restored._cfg == stream._cfg


def test_save_appends_npz_suffix():
    """`save` must write a readable file even when path has no .npz suffix."""
    import pathlib
    import tempfile

    prices, mu, cfg, _assets = _make_prices_mu()
    stream = BasanosStream.from_warmup(prices.head(50), mu.head(50), cfg)

    with tempfile.TemporaryDirectory() as tmp:
        p_no_suffix = pathlib.Path(tmp) / "stream"
        stream.save(p_no_suffix)
        # numpy appends .npz when the suffix is absent
        restored = BasanosStream.load(pathlib.Path(tmp) / "stream.npz")

    assert restored.assets == stream.assets


def test_save_writes_format_version():
    """`save` must embed a ``format_version`` key in the archive."""
    import pathlib
    import tempfile

    prices, mu, cfg, _assets = _make_prices_mu()
    stream = BasanosStream.from_warmup(prices.head(50), mu.head(50), cfg)

    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "stream.npz"
        stream.save(p)
        data = np.load(p, allow_pickle=False)

    assert "format_version" in data
    assert int(data["format_version"]) == 2


def test_load_raises_on_missing_format_version():
    """`load` must raise ValueError when the archive has no format_version key."""
    import pathlib
    import tempfile

    prices, mu, cfg, _assets = _make_prices_mu()
    stream = BasanosStream.from_warmup(prices.head(50), mu.head(50), cfg)

    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "stream.npz"
        stream.save(p)

        # Re-save without the format_version key to simulate a legacy archive.
        data = dict(np.load(p, allow_pickle=False))
        del data["format_version"]
        np.savez(p, **data)

        with pytest.raises(ValueError, match="format version tag"):
            BasanosStream.load(p)


def test_load_raises_on_wrong_format_version():
    """`load` must raise ValueError when the stored version does not match."""
    import pathlib
    import tempfile

    prices, mu, cfg, _assets = _make_prices_mu()
    stream = BasanosStream.from_warmup(prices.head(50), mu.head(50), cfg)

    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "stream.npz"
        stream.save(p)

        # Overwrite format_version with a future version number.
        data = dict(np.load(p, allow_pickle=False))
        data["format_version"] = np.array(999)
        np.savez(p, **data)

        with pytest.raises(ValueError, match="format version 999"):
            BasanosStream.load(p)


# ─── SlidingWindowConfig streaming tests ─────────────────────────────────────


def _make_sw_stream(warmup_len: int = 50, window: int = 20, n_factors: int = 2):
    """Return (stream, prices_np, mu_np, prices, mu, assets) for SW integration tests."""
    prices, mu, _, assets = _make_prices_mu(n_total=80)
    cfg_sw = BasanosConfig(
        vola=5,
        corr=10,
        clip=3.0,
        shrink=0.5,
        aum=1e6,
        covariance_config=SlidingWindowConfig(window=window, n_factors=n_factors),
    )
    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg_sw)
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()
    return stream, prices_np, mu_np, prices, mu, assets


def test_sw_step_returns_step_result():
    """step() with SlidingWindowConfig must return a StepResult."""
    stream, prices_np, mu_np, prices, _mu, _assets = _make_sw_stream()
    warmup_len = 50
    result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])
    assert isinstance(result, StepResult)


def test_sw_step_cash_position_and_vola_shape():
    """cash_position and vola must have shape (N,) for SW mode."""
    stream, prices_np, mu_np, prices, _mu, assets = _make_sw_stream()
    warmup_len = 50
    result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])
    assert result.cash_position.shape == (len(assets),)
    assert result.vola.shape == (len(assets),)


def test_sw_step_valid_status_after_full_warmup():
    """step() must return 'valid' when the buffer is full and signal is non-zero."""
    stream, prices_np, mu_np, prices, _mu, _assets = _make_sw_stream(warmup_len=50, window=20)
    warmup_len = 50
    result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])
    assert result.status in {"valid", "degenerate"}


def test_sw_warmup_status_when_buffer_not_full():
    """step() must return 'warmup' when fewer than window rows have been seen."""
    prices, mu, _, assets = _make_prices_mu(n_total=80)
    # warmup batch shorter than window: stream starts in warmup
    cfg_sw = BasanosConfig(
        vola=5,
        corr=10,
        clip=3.0,
        shrink=0.5,
        aum=1e6,
        covariance_config=SlidingWindowConfig(window=20, n_factors=2),
    )
    # warmup_len=15 < window=20 → buffer is padded; first step still in warmup
    stream = BasanosStream.from_warmup(prices.head(15), mu.head(15), cfg_sw)
    assert stream._state.sw_ret_buf is not None
    assert stream._state.sw_ret_buf.shape == (20, len(assets))

    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()
    result = stream.step(prices_np[15], mu_np[15], prices["date"][15])
    assert result.status == "warmup"


def test_sw_short_warmup_all_nan_steps_return_warmup_then_transition():
    """All window-n_rows steps return 'warmup'; the next step is non-warmup.

    When the warmup batch has fewer rows than the sliding window, the rolling
    buffer contains NaN-padded prefix rows.  Each step shifts one NaN row out
    and appends a real row.  The buffer is fully populated with real data
    exactly when step_count reaches window — the same point in_warmup becomes
    False.  This test asserts that:

    1. All ``window - n_rows`` warmup steps return ``status="warmup"``.
    2. Cash positions are NaN during that period.
    3. The first post-warmup step returns a non-warmup status.
    4. After the transition the buffer contains no NaN rows.
    """
    n_total = 80
    window = 20
    n_rows = 15  # warmup batch shorter than window by 5 rows
    prices, mu, _, assets = _make_prices_mu(n_total=n_total)
    cfg_sw = BasanosConfig(
        vola=5,
        corr=10,
        clip=3.0,
        shrink=0.5,
        aum=1e6,
        covariance_config=SlidingWindowConfig(window=window, n_factors=2),
    )
    stream = BasanosStream.from_warmup(prices.head(n_rows), mu.head(n_rows), cfg_sw)
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    nan_pad_steps = window - n_rows  # 5 steps should all be "warmup"

    for i in range(nan_pad_steps):
        result = stream.step(prices_np[n_rows + i], mu_np[n_rows + i], prices["date"][n_rows + i])
        assert result.status == "warmup", f"step {i + 1}: expected 'warmup', got {result.status!r}"
        assert np.all(np.isnan(result.cash_position)), f"step {i + 1}: expected NaN cash_position during warmup"

    # First post-warmup step: buffer is now fully populated with real data.
    result_post = stream.step(
        prices_np[n_rows + nan_pad_steps],
        mu_np[n_rows + nan_pad_steps],
        prices["date"][n_rows + nan_pad_steps],
    )
    assert result_post.status != "warmup", f"expected non-warmup after buffer full, got {result_post.status!r}"
    assert stream._state.sw_ret_buf is not None
    assert np.all(np.isfinite(stream._state.sw_ret_buf)), "buffer still contains NaN rows after warmup ends"


def test_sw_short_warmup_save_load_preserves_warmup_period():
    """A stream saved mid-NaN-padding period restores the correct warmup state."""
    import pathlib
    import tempfile

    n_total = 80
    window = 20
    n_rows = 15
    prices, mu, _, assets = _make_prices_mu(n_total=n_total)
    cfg_sw = BasanosConfig(
        vola=5,
        corr=10,
        clip=3.0,
        shrink=0.5,
        aum=1e6,
        covariance_config=SlidingWindowConfig(window=window, n_factors=2),
    )
    stream = BasanosStream.from_warmup(prices.head(n_rows), mu.head(n_rows), cfg_sw)
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "sw_short.npz"
        stream.save(p)
        restored = BasanosStream.load(p)

    # Both the original and restored stream should return identical results
    # for each remaining warmup step.
    for i in range(window - n_rows + 1):
        idx = n_rows + i
        r_orig = stream.step(prices_np[idx], mu_np[idx], prices["date"][idx])
        r_rest = restored.step(prices_np[idx], mu_np[idx], prices["date"][idx])
        assert r_orig.status == r_rest.status, f"step {i + 1}: status mismatch"
        np.testing.assert_array_equal(r_orig.cash_position, r_rest.cash_position)


def test_sw_step_zero_signal_yields_zero_signal():
    """step() must return 'zero_signal' when mu is all-zero in SW mode."""
    stream, prices_np, mu_np, prices, _mu, _assets = _make_sw_stream(warmup_len=50, window=20)
    warmup_len = 50
    zero_mu = np.zeros_like(mu_np[warmup_len])
    result = stream.step(prices_np[warmup_len], zero_mu, prices["date"][warmup_len])
    assert result.status == "zero_signal"
    assert np.all(result.cash_position[np.isfinite(result.cash_position)] == 0.0)


def test_sw_step_all_nan_prices_yields_degenerate():
    """step() must return 'degenerate' when all prices are NaN in SW mode."""
    stream, prices_np, mu_np, prices, _mu, _assets = _make_sw_stream(warmup_len=50, window=20)
    warmup_len = 50
    nan_prices = np.full_like(prices_np[warmup_len], np.nan)
    result = stream.step(nan_prices, mu_np[warmup_len], prices["date"][warmup_len])
    assert result.status == "degenerate"


def test_sw_step_singular_woodbury_yields_degenerate():
    """step() must return 'degenerate' when the Woodbury solve raises SingularMatrixError."""
    stream, prices_np, mu_np, prices, _mu, _assets = _make_sw_stream(warmup_len=50, window=20)
    warmup_len = 50
    with patch("basanos.math._stream.FactorModel.from_returns") as mock_fm:
        mock_fm.side_effect = ValueError("singular")
        result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])
    assert result.status == "degenerate"


def test_sw_step_degenerate_denom_yields_degenerate():
    """step() must return 'degenerate' when the Woodbury denominator is non-finite."""
    stream, prices_np, mu_np, prices, _mu, _assets = _make_sw_stream(warmup_len=50, window=20)
    warmup_len = 50
    with patch("basanos.math._stream.FactorModel.solve", return_value=np.zeros(3)):
        result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])
    assert result.status == "degenerate"


def test_sw_step_fm_solve_raises_yields_degenerate():
    """step() must return 'degenerate' when fm.solve() raises SingularMatrixError."""
    from basanos.exceptions import SingularMatrixError

    stream, prices_np, mu_np, prices, _mu, _assets = _make_sw_stream(warmup_len=50, window=20)
    warmup_len = 50
    with patch("basanos.math._stream.FactorModel.solve", side_effect=SingularMatrixError("singular")):
        result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])
    assert result.status == "degenerate"


def test_sw_step_matches_batch_engine():
    """SW stream step at warmup_len must match the batch engine cash_position at that row."""
    prices, mu, _, assets = _make_prices_mu(n_total=80)
    warmup_len = 50
    window = 20
    cfg_sw = BasanosConfig(
        vola=5,
        corr=10,
        clip=3.0,
        shrink=0.5,
        aum=1e6,
        covariance_config=SlidingWindowConfig(window=window, n_factors=2),
    )
    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg_sw)
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    result = stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])

    # Batch engine on prices[:warmup_len+1] should agree at the last row
    from basanos.math import BasanosEngine

    engine = BasanosEngine(prices=prices.head(warmup_len + 1), mu=mu.head(warmup_len + 1), cfg=cfg_sw)
    batch_pos = engine.cash_position.select(assets).to_numpy()[-1]

    np.testing.assert_allclose(result.cash_position, batch_pos, rtol=1e-6, equal_nan=True)


def test_sw_save_load_roundtrip():
    """A SW stream restored from disk must produce bit-for-bit identical step() output."""
    import pathlib
    import tempfile

    stream, prices_np, mu_np, prices, _mu, _assets = _make_sw_stream()
    warmup_len = 50

    with tempfile.TemporaryDirectory() as tmp:
        p = pathlib.Path(tmp) / "sw_stream.npz"
        stream.save(p)
        restored = BasanosStream.load(p)

    step_date = prices["date"][warmup_len]
    result_orig = stream.step(prices_np[warmup_len], mu_np[warmup_len], step_date)
    result_loaded = restored.step(prices_np[warmup_len], mu_np[warmup_len], step_date)

    np.testing.assert_array_equal(result_orig.cash_position, result_loaded.cash_position)
    assert result_orig.status == result_loaded.status


def test_sw_max_components_caps_k_eff():
    """max_components must cap the effective SVD rank passed to FactorModel.from_returns."""
    from basanos.math._factor_model import FactorModel

    prices, mu, _, assets = _make_prices_mu(n_total=80)
    warmup_len = 50
    window = 20

    cfg = BasanosConfig(
        vola=5,
        corr=10,
        clip=3.0,
        shrink=0.5,
        aum=1e6,
        covariance_config=SlidingWindowConfig(window=window, n_factors=10, max_components=2),
    )
    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)

    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    captured_k: list[int] = []
    original_from_returns = FactorModel.from_returns

    def capture_k(returns, k):
        captured_k.append(k)
        return original_from_returns(returns, k)

    with patch("basanos.math._stream.FactorModel.from_returns", side_effect=capture_k):
        stream.step(prices_np[warmup_len], mu_np[warmup_len], prices["date"][warmup_len])

    assert len(captured_k) == 1, "from_returns should be called exactly once"
    assert captured_k[0] <= 2, f"k_eff must be capped at max_components=2, got {captured_k[0]}"


def test_sw_max_components_none_equals_no_cap():
    """max_components=None must produce the same result as omitting max_components."""
    prices, mu, _, assets = _make_prices_mu(n_total=80)
    warmup_len = 50
    window = 20

    cfg_explicit_none = BasanosConfig(
        vola=5,
        corr=10,
        clip=3.0,
        shrink=0.5,
        aum=1e6,
        covariance_config=SlidingWindowConfig(window=window, n_factors=2, max_components=None),
    )
    cfg_default = BasanosConfig(
        vola=5,
        corr=10,
        clip=3.0,
        shrink=0.5,
        aum=1e6,
        covariance_config=SlidingWindowConfig(window=window, n_factors=2),
    )

    stream_explicit = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg_explicit_none)
    stream_default = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg_default)

    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()
    step_date = prices["date"][warmup_len]

    result_explicit = stream_explicit.step(prices_np[warmup_len], mu_np[warmup_len], step_date)
    result_default = stream_default.step(prices_np[warmup_len], mu_np[warmup_len], step_date)

    np.testing.assert_array_equal(result_explicit.cash_position, result_default.cash_position)
    assert result_explicit.status == result_default.status


# ─── max_turnover constraint in BasanosStream.step() ─────────────────────────


def _make_stream_for_turnover_test(
    max_turnover: float | None = None,
    n_total: int = 80,
    n_assets: int = 5,
    seed: int = 7,
    warmup_len: int = 50,
) -> tuple[BasanosStream, np.ndarray, np.ndarray]:
    """Return (stream, prices_np, mu_np) for turnover constraint tests."""
    prices, mu, _, assets = _make_prices_mu(n_total=n_total, n_assets=n_assets, seed=seed)
    cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6, max_turnover=max_turnover)
    stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()
    return stream, prices_np, mu_np


def test_step_max_turnover_none_unchanged():
    """step() with max_turnover=None must produce the same positions as no constraint."""
    prices, mu, _, assets = _make_prices_mu(n_total=80, n_assets=5, seed=7)
    cfg_none = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6)
    cfg_explicit = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6, max_turnover=None)
    warmup_len = 50

    stream_none = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg_none)
    stream_explicit = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg_explicit)

    prices_np = prices.select(assets).to_numpy()
    mu_np = mu.select(assets).to_numpy()

    for i in range(warmup_len, len(prices_np)):
        r_none = stream_none.step(prices_np[i], mu_np[i])
        r_explicit = stream_explicit.step(prices_np[i], mu_np[i])
        np.testing.assert_array_equal(r_none.cash_position, r_explicit.cash_position)


def test_step_max_turnover_constrains_l1_position_change():
    """step() with max_turnover set must cap sum(|Δx_t|) at every non-warmup step."""
    max_turnover = 1e4
    stream, prices_np, mu_np = _make_stream_for_turnover_test(max_turnover=max_turnover)
    warmup_len = 50

    prev_pos = np.zeros(prices_np.shape[1])
    for i in range(warmup_len, len(prices_np)):
        result = stream.step(prices_np[i], mu_np[i])
        curr = np.nan_to_num(result.cash_position, nan=0.0)
        delta_l1 = float(np.sum(np.abs(curr - prev_pos)))
        assert delta_l1 <= max_turnover + 1e-6, f"step {i}: turnover {delta_l1:.4f} exceeds max_turnover {max_turnover}"
        prev_pos = curr


def test_step_tight_max_turnover_reduces_total_turnover():
    """A tighter max_turnover in step() must produce less or equal total L1 turnover."""
    n_total, warmup_len = 80, 50
    prices, mu, _, assets = _make_prices_mu(n_total=n_total, n_assets=5, seed=9)

    def collect_positions(max_turnover: float | None) -> np.ndarray:
        cfg = BasanosConfig(vola=5, corr=10, clip=3.0, shrink=0.5, aum=1e6, max_turnover=max_turnover)
        stream = BasanosStream.from_warmup(prices.head(warmup_len), mu.head(warmup_len), cfg)
        prices_np = prices.select(assets).to_numpy()
        mu_np = mu.select(assets).to_numpy()
        rows = [stream.step(prices_np[i], mu_np[i]).cash_position for i in range(warmup_len, n_total)]
        return np.array(rows)

    pos_unconstrained = collect_positions(None)
    pos_constrained = collect_positions(5e3)

    def total_l1(pos: np.ndarray) -> float:
        total = 0.0
        for i in range(1, len(pos)):
            prev = np.nan_to_num(pos[i - 1], nan=0.0)
            curr = np.nan_to_num(pos[i], nan=0.0)
            total += float(np.sum(np.abs(curr - prev)))
        return total

    assert total_l1(pos_constrained) <= total_l1(pos_unconstrained) + 1e-6, (
        "Constrained stream produced more total turnover than unconstrained stream"
    )
