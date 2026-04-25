"""Tests for basanos.analytics shim.

Verifies that every public name re-exported by basanos.analytics resolves to
the same object as the original in jquantstats, so downstream code that does
`from basanos.analytics import Portfolio` keeps working as jquantstats evolves.
"""

import importlib

import pytest

# ─── import surface ───────────────────────────────────────────────────────────


def test_analytics_module_is_importable():
    """basanos.analytics can be imported without error."""
    importlib.import_module("basanos.analytics")


@pytest.mark.parametrize("name", ["NativeFrame", "NativeFrameOrScalar", "Portfolio"])
def test_name_importable_from_analytics(name):
    """Each public name is accessible as an attribute of basanos.analytics."""
    mod = importlib.import_module("basanos.analytics")
    assert hasattr(mod, name), f"basanos.analytics has no attribute '{name}'"


# ─── identity with jquantstats ────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["NativeFrame", "NativeFrameOrScalar", "Portfolio"])
def test_shim_is_same_object_as_jquantstats(name):
    """Re-exported names must be the exact same objects as in jquantstats."""
    analytics = importlib.import_module("basanos.analytics")
    jqs = importlib.import_module("jquantstats")
    assert getattr(analytics, name) is getattr(jqs, name)


def test_all_lists_expected_names():
    """__all__ must match exactly the three forwarded names."""
    from basanos import analytics

    assert set(analytics.__all__) == {"NativeFrame", "NativeFrameOrScalar", "Portfolio"}


# ─── Portfolio usable via shim ────────────────────────────────────────────────


def test_portfolio_via_shim_matches_conftest_fixture(portfolio):
    """Portfolio imported from the shim must be the same class as the fixture."""
    from basanos.analytics import Portfolio

    assert isinstance(portfolio, Portfolio)
