"""Mathematics subpackage for basanos.

Provides correlation-aware optimization utilities and related helpers.
This package exposes a minimal, stable public API while keeping internal
implementation details in private modules so documentation remains clean
and links are hierarchical in pdoc.

Public API:
- basanos.math.optimizer

Private modules (subject to change):
- basanos.math._linalg
- basanos.math._signal

Usage:
    Prefer importing public classes from this package namespace:

    >>> from basanos.math import BasanosConfig, BasanosEngine  # doctest: +SKIP
    >>> from basanos.math import BasanosConfig, BasanosEngine
    >>> issubclass(BasanosConfig, object) and issubclass(BasanosEngine, object)
    True

Notes:
    These re-exports are stable; private modules are implementation details
    and may change without notice.
"""

# Public re-exports (explicit aliases so linters recognize intent)
from .optimizer import BasanosConfig as BasanosConfig
from .optimizer import BasanosEngine as BasanosEngine
