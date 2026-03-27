"""basanos.

Examples:
    >>> import basanos
    >>> basanos.__name__
    'basanos'
"""

from basanos._deprecation import warn_deprecated as warn_deprecated
from basanos._logging import JSONFormatter as JSONFormatter
from basanos.exceptions import (
    BasanosError as BasanosError,
)
from basanos.exceptions import (
    ColumnMismatchError as ColumnMismatchError,
)
from basanos.exceptions import (
    DimensionMismatchError as DimensionMismatchError,
)
from basanos.exceptions import (
    ExcessiveNullsError as ExcessiveNullsError,
)
from basanos.exceptions import (
    FactorModelError as FactorModelError,
)
from basanos.exceptions import (
    InsufficientDataError as InsufficientDataError,
)
from basanos.exceptions import (
    MissingDateColumnError as MissingDateColumnError,
)
from basanos.exceptions import (
    MonotonicPricesError as MonotonicPricesError,
)
from basanos.exceptions import (
    NonPositivePricesError as NonPositivePricesError,
)
from basanos.exceptions import (
    NonSquareMatrixError as NonSquareMatrixError,
)
from basanos.exceptions import (
    ShapeMismatchError as ShapeMismatchError,
)
from basanos.exceptions import (
    SingularMatrixError as SingularMatrixError,
)

__all__ = [
    "BasanosError",
    "ColumnMismatchError",
    "DimensionMismatchError",
    "ExcessiveNullsError",
    "FactorModelError",
    "InsufficientDataError",
    "JSONFormatter",
    "MissingDateColumnError",
    "MonotonicPricesError",
    "NonPositivePricesError",
    "NonSquareMatrixError",
    "ShapeMismatchError",
    "SingularMatrixError",
    "warn_deprecated",
]
