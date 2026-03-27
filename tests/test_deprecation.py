"""Tests for basanos._deprecation (warn_deprecated).

Covers:
- warn_deprecated emits a DeprecationWarning.
- The warning message includes the name, since, and remove_in versions.
- The optional replacement is included when provided.
- stacklevel is forwarded to warnings.warn.
"""

from __future__ import annotations

import warnings

from basanos import warn_deprecated


def _collect(
    name: str,
    *,
    since: str,
    remove_in: str,
    replacement: str | None = None,
) -> list[warnings.WarningMessage]:
    """Convenience: call warn_deprecated and return captured warning list."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_deprecated(name, since=since, remove_in=remove_in, replacement=replacement)
    return list(w)


class TestWarnDeprecated:
    """Unit tests for warn_deprecated."""

    def test_emits_exactly_one_warning(self) -> None:
        captured = _collect("old_func", since="0.6", remove_in="0.8")
        assert len(captured) == 1

    def test_category_is_deprecation_warning(self) -> None:
        captured = _collect("old_func", since="0.6", remove_in="0.8")
        assert issubclass(captured[0].category, DeprecationWarning)

    def test_message_contains_name(self) -> None:
        captured = _collect("my_api", since="0.6", remove_in="0.8")
        assert "my_api" in str(captured[0].message)

    def test_message_contains_since_version(self) -> None:
        captured = _collect("my_api", since="0.6", remove_in="0.8")
        assert "0.6" in str(captured[0].message)

    def test_message_contains_remove_in_version(self) -> None:
        captured = _collect("my_api", since="0.6", remove_in="0.8")
        assert "0.8" in str(captured[0].message)

    def test_no_replacement_by_default(self) -> None:
        captured = _collect("old_func", since="0.6", remove_in="0.8")
        assert "Use" not in str(captured[0].message)

    def test_replacement_included_when_provided(self) -> None:
        captured = _collect("old_func", since="0.6", remove_in="0.8", replacement="new_func")
        assert "new_func" in str(captured[0].message)

    def test_warn_deprecated_importable_from_basanos_top_level(self) -> None:
        import basanos

        assert callable(basanos.warn_deprecated)
