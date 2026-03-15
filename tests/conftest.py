"""Fixtures for tests.

Security note: Test code uses pytest assertions (S101), which are intentional
and safe in the test context. No subprocess calls (S603/S607) are used here.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def resource_dir():
    """Fixture that provides the path to the test resources directory."""
    return Path(__file__).parent / "resources"
