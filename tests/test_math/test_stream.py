"""Tests for basanos.math._stream.StepResult.

Tests cover:
- Direct construction and field access
- Frozen (immutable) contract
- Public export from basanos.math
"""

import dataclasses

import numpy as np
import pytest

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
