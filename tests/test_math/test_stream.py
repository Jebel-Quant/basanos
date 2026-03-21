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

"""

from __future__ import annotations
import dataclasses

import numpy as np
import pytest

from basanos.math._stream import _StreamState

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
        profit_variance=0.0,
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
    assert isinstance(s.profit_variance, float)
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


def test_profit_variance_mutable() -> None:
    """profit_variance scalar must be mutable."""
    s = _make_state(2)
    s.profit_variance = 1.23
    assert s.profit_variance == pytest.approx(1.23)


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
        "profit_variance",
        "prev_price",
        "prev_cash_pos",
        "step_count",
    }
    assert fields == expected
from basanos.math import StepResult
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
