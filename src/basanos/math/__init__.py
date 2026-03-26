"""Mathematics subpackage for basanos.

Provides correlation-aware optimization utilities and related helpers.
This package exposes a minimal, stable public API while keeping internal
implementation details in private modules so documentation remains clean.

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
from ._engine_solve import MatrixBundle as MatrixBundle
from ._engine_solve import SolveStatus as SolveStatus
from ._engine_solve import WarmupState as WarmupState
from ._factor_model import FactorModel as FactorModel
from ._stream import BasanosStream as BasanosStream
from ._stream import StepResult as StepResult
from .optimizer import BasanosConfig as BasanosConfig
from .optimizer import BasanosEngine as BasanosEngine
from .optimizer import CovarianceConfig as CovarianceConfig
from .optimizer import CovarianceMode as CovarianceMode
from .optimizer import EwmaShrinkConfig as EwmaShrinkConfig
from .optimizer import SlidingWindowConfig as SlidingWindowConfig

__all__ = [
    "BasanosConfig",
    "BasanosEngine",
    "BasanosStream",
    "CovarianceConfig",
    "CovarianceMode",
    "EwmaShrinkConfig",
    "FactorModel",
    "MatrixBundle",
    "SlidingWindowConfig",
    "SolveStatus",
    "StepResult",
    "WarmupState",
]
