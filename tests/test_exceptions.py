"""Tests for basanos.exceptions."""

from __future__ import annotations

import pytest

from basanos.exceptions import InsufficientDataError


def test_insufficient_data_error_default_message() -> None:
    """InsufficientDataError with no detail uses the built-in message."""
    err = InsufficientDataError()
    assert "Insufficient finite data" in str(err)


def test_insufficient_data_error_custom_detail() -> None:
    """InsufficientDataError with a detail string uses that string as the message."""
    err = InsufficientDataError("All diagonal entries are non-finite.")
    assert str(err) == "All diagonal entries are non-finite."


def test_insufficient_data_error_is_value_error() -> None:
    """InsufficientDataError must be catchable as ValueError."""
    with pytest.raises(ValueError, match="Insufficient finite data"):
        raise InsufficientDataError()
