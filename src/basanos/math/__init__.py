"""Mathematics subpackage for basanos.

Provides correlation-aware optimization utilities and related helpers.
This package exposes a minimal, stable public API while keeping internal
implementation details in private modules so documentation remains clean
and links are hierarchical in pdoc.

Public API:
- basanos.math.taipan

Private modules (subject to change):
- basanos.math._linalg
- basanos.math._signal

Usage:
    Prefer importing public classes from this package namespace:

    >>> from basanos.math import TaipanConfig, TaipanEngine  # doctest: +SKIP

Notes:
    These re-exports are stable; private modules are implementation details
    and may change without notice.
"""

# Public re-exports (explicit aliases so linters recognize intent)
from .taipan import TaipanConfig as TaipanConfig
from .taipan import TaipanEngine as TaipanEngine
