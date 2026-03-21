"""Tests for basanos.math._stream.StepResult.

Tests cover:
- Direct construction and field access
- Frozen (immutable) contract
- Public export from basanos.math
- BasanosStream.from_warmup() acceptance criterion
- BasanosStream.step() correctness and error handling
"""

import dataclasses
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from basanos.exceptions import MissingDateColumnError
from basanos.math import (
    BasanosConfig,
    BasanosEngine,
    BasanosStream,
    SlidingWindowConfig,
    StepResult,
    _StreamState,
)
from basanos.math._stream import StepResult as StepResultDirect

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


def test_is_dataclass():
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


def test_stream_state_exported():
    """_StreamState must be importable from basanos.math."""
    assert _StreamState is not None


def test_stream_state_is_dataclass():
    """_StreamState must be a dataclass."""
    assert dataclasses.is_dataclass(_StreamState)


# ─── from_warmup validation ───────────────────────────────────────────────────


def test_from_warmup_rejects_sliding_window():
    """from_warmup raises TypeError for SlidingWindowConfig."""
    prices, mu, _, _ = _make_prices_mu()
    cfg_sw = BasanosConfig(
        vola=5,
        corr=10,
        clip=3.0,
        shrink=0.5,
        aum=1e6,
        covariance_config=SlidingWindowConfig(window=30, n_factors=2),
    )
    with pytest.raises(TypeError, match="EwmaShrinkConfig"):
        BasanosStream.from_warmup(prices.head(40), mu.head(40), cfg_sw)


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
