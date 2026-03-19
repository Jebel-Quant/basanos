"""Domain-specific exception types for the Basanos package.

This module defines a hierarchy of exceptions that provide meaningful context
when linear-algebra or data-validation errors occur within the library.

All exceptions inherit from :class:`BasanosError` so callers can catch the
entire family with a single ``except BasanosError`` clause if they prefer.

Examples:
    >>> raise NonSquareMatrixError(3, 2)
    Traceback (most recent call last):
        ...
    basanos.exceptions.NonSquareMatrixError: Matrix must be square, got shape (3, 2).
"""

from __future__ import annotations


class BasanosError(Exception):
    """Base class for all Basanos domain errors."""


class NonSquareMatrixError(BasanosError, ValueError):
    """Raised when a matrix is required to be square but is not.

    Args:
        rows: Number of rows in the offending matrix.
        cols: Number of columns in the offending matrix.

    Examples:
        >>> raise NonSquareMatrixError(3, 2)
        Traceback (most recent call last):
            ...
        basanos.exceptions.NonSquareMatrixError: Matrix must be square, got shape (3, 2).
    """

    def __init__(self, rows: int, cols: int) -> None:
        """Initialize with the offending matrix shape."""
        super().__init__(f"Matrix must be square, got shape ({rows}, {cols}).")
        self.rows = rows
        self.cols = cols


class DimensionMismatchError(BasanosError, ValueError):
    """Raised when vector and matrix dimensions are incompatible.

    Args:
        vector_size: Length of the offending vector.
        matrix_size: Expected dimension inferred from the matrix.

    Examples:
        >>> raise DimensionMismatchError(3, 2)
        Traceback (most recent call last):
            ...
        basanos.exceptions.DimensionMismatchError: Vector length 3 does not match matrix dimension 2.
    """

    def __init__(self, vector_size: int, matrix_size: int) -> None:
        """Initialize with the offending vector and matrix sizes."""
        super().__init__(f"Vector length {vector_size} does not match matrix dimension {matrix_size}.")
        self.vector_size = vector_size
        self.matrix_size = matrix_size


class SingularMatrixError(BasanosError, ValueError):
    """Raised when a matrix is (numerically) singular and cannot be inverted.

    This wraps :class:`numpy.linalg.LinAlgError` to provide domain-specific
    context.

    Examples:
        >>> raise SingularMatrixError()
        Traceback (most recent call last):
            ...
        basanos.exceptions.SingularMatrixError: Matrix is singular and cannot be solved.
    """

    def __init__(self, detail: str = "") -> None:
        """Initialize with an optional extra detail string."""
        msg = "Matrix is singular and cannot be solved."
        if detail:
            msg = f"{msg} {detail}"
        super().__init__(msg)


class InsufficientDataError(BasanosError, ValueError):
    """Raised when there are too few finite entries to perform a computation.

    Examples:
        >>> raise InsufficientDataError("All diagonal entries are non-finite.")
        Traceback (most recent call last):
            ...
        basanos.exceptions.InsufficientDataError: All diagonal entries are non-finite.
    """

    def __init__(self, detail: str = "") -> None:
        """Initialize with an optional detail message that overrides the default."""
        msg = "Insufficient finite data to complete the computation."
        if detail:
            msg = detail
        super().__init__(msg)


class MissingDateColumnError(BasanosError, ValueError):
    """Raised when a required ``'date'`` column is absent from a DataFrame.

    Args:
        frame_name: Descriptive name of the frame missing the column (e.g. ``"prices"``).

    Examples:
        >>> raise MissingDateColumnError("prices")
        Traceback (most recent call last):
            ...
        basanos.exceptions.MissingDateColumnError: DataFrame 'prices' is missing the required 'date' column.
    """

    def __init__(self, frame_name: str) -> None:
        """Initialize with the name of the frame that is missing the column."""
        super().__init__(f"DataFrame '{frame_name}' is missing the required 'date' column.")
        self.frame_name = frame_name


class ShapeMismatchError(BasanosError, ValueError):
    """Raised when two DataFrames have incompatible shapes.

    Args:
        prices_shape: Shape of the prices DataFrame.
        mu_shape: Shape of the mu DataFrame.

    Examples:
        >>> raise ShapeMismatchError((10, 3), (9, 3))
        Traceback (most recent call last):
            ...
        basanos.exceptions.ShapeMismatchError: 'prices' and 'mu' must have the same shape, got (10, 3) vs (9, 3).
    """

    def __init__(self, prices_shape: tuple[int, int], mu_shape: tuple[int, int]) -> None:
        """Initialize with the shapes of the two mismatched DataFrames."""
        super().__init__(f"'prices' and 'mu' must have the same shape, got {prices_shape} vs {mu_shape}.")
        self.prices_shape = prices_shape
        self.mu_shape = mu_shape


class ColumnMismatchError(BasanosError, ValueError):
    """Raised when two DataFrames have different column sets.

    Args:
        prices_columns: Columns of the prices DataFrame.
        mu_columns: Columns of the mu DataFrame.

    Examples:
        >>> raise ColumnMismatchError(["A", "B"], ["A", "C"])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        basanos.exceptions.ColumnMismatchError: 'prices' and 'mu' must have identical columns...
    """

    def __init__(self, prices_columns: list[str], mu_columns: list[str]) -> None:
        """Initialize with the column lists of the two mismatched DataFrames."""
        super().__init__(
            f"'prices' and 'mu' must have identical columns; got {sorted(prices_columns)} vs {sorted(mu_columns)}."
        )
        self.prices_columns = prices_columns
        self.mu_columns = mu_columns


class NonPositivePricesError(BasanosError, ValueError):
    """Raised when an asset column contains zero or negative prices.

    Log-return computation requires strictly positive prices.

    Args:
        asset: Name of the asset with the offending values.

    Examples:
        >>> raise NonPositivePricesError("A")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        basanos.exceptions.NonPositivePricesError: Asset 'A' contains non-positive...
    """

    def __init__(self, asset: str) -> None:
        """Initialize with the name of the asset that contains non-positive prices."""
        super().__init__(f"Asset '{asset}' contains non-positive prices; strictly positive values are required.")
        self.asset = asset


class ExcessiveNullsError(BasanosError, ValueError):
    """Raised when an asset column contains too many null values.

    Args:
        asset: Name of the offending asset column.
        null_fraction: Observed fraction of null values (0.0 to 1.0).
        max_fraction: Maximum allowed fraction of null values.

    Examples:
        >>> raise ExcessiveNullsError("A", 1.0, 0.9)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        basanos.exceptions.ExcessiveNullsError: Asset 'A' has 100% null values,...
    """

    def __init__(self, asset: str, null_fraction: float, max_fraction: float) -> None:
        """Initialize with the asset name and the observed/maximum null fractions."""
        super().__init__(
            f"Asset '{asset}' has {null_fraction:.0%} null values, "
            f"exceeding the maximum allowed fraction of {max_fraction:.0%}."
        )
        self.asset = asset
        self.null_fraction = null_fraction
        self.max_fraction = max_fraction


class IntegerIndexBoundError(BasanosError, TypeError):
    """Raised when a row-index bound is not an integer.

    Args:
        param: Name of the offending parameter (e.g. ``"start"`` or ``"end"``).
        actual_type: The ``type.__name__`` of the value that was supplied.

    Examples:
        >>> raise IntegerIndexBoundError("start", "str")
        Traceback (most recent call last):
            ...
        basanos.exceptions.IntegerIndexBoundError: start must be an integer, got str.
    """

    def __init__(self, param: str, actual_type: str) -> None:
        """Initialize with the parameter name and the offending type."""
        super().__init__(f"{param} must be an integer, got {actual_type}.")
        self.param = param
        self.actual_type = actual_type


class InvalidPricesTypeError(BasanosError, TypeError):
    """Raised when ``prices`` is not a :class:`polars.DataFrame`.

    Args:
        actual_type: The ``type.__name__`` of the value that was supplied.

    Examples:
        >>> raise InvalidPricesTypeError("list")
        Traceback (most recent call last):
            ...
        basanos.exceptions.InvalidPricesTypeError: prices must be pl.DataFrame, got list.
    """

    def __init__(self, actual_type: str) -> None:
        """Initialize with the offending type name."""
        super().__init__(f"prices must be pl.DataFrame, got {actual_type}.")
        self.actual_type = actual_type


class InvalidCashPositionTypeError(BasanosError, TypeError):
    """Raised when ``cashposition`` is not a :class:`polars.DataFrame`.

    Args:
        actual_type: The ``type.__name__`` of the value that was supplied.

    Examples:
        >>> raise InvalidCashPositionTypeError("dict")
        Traceback (most recent call last):
            ...
        basanos.exceptions.InvalidCashPositionTypeError: cashposition must be pl.DataFrame, got dict.
    """

    def __init__(self, actual_type: str) -> None:
        """Initialize with the offending type name."""
        super().__init__(f"cashposition must be pl.DataFrame, got {actual_type}.")
        self.actual_type = actual_type


class RowCountMismatchError(BasanosError, ValueError):
    """Raised when ``prices`` and ``cashposition`` have different numbers of rows.

    Args:
        prices_rows: Number of rows in the prices DataFrame.
        cashposition_rows: Number of rows in the cashposition DataFrame.

    Examples:
        >>> raise RowCountMismatchError(10, 9)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        basanos.exceptions.RowCountMismatchError: cashposition and prices must have the same number of rows...
    """

    def __init__(self, prices_rows: int, cashposition_rows: int) -> None:
        """Initialize with the row counts of the two mismatched DataFrames."""
        super().__init__(
            f"cashposition and prices must have the same number of rows, "
            f"got cashposition={cashposition_rows} and prices={prices_rows}."
        )
        self.prices_rows = prices_rows
        self.cashposition_rows = cashposition_rows


class NonPositiveAumError(BasanosError, ValueError):
    """Raised when ``aum`` is not strictly positive.

    Args:
        aum: The non-positive value that was supplied.

    Examples:
        >>> raise NonPositiveAumError(0.0)
        Traceback (most recent call last):
            ...
        basanos.exceptions.NonPositiveAumError: aum must be strictly positive, got 0.0.
    """

    def __init__(self, aum: float) -> None:
        """Initialize with the offending aum value."""
        super().__init__(f"aum must be strictly positive, got {aum}.")
        self.aum = aum


class NullsAfterCleaningError(BasanosError, ValueError):
    """Raised when a profit column still contains nulls after the cleaning pass.

    This is an internal invariant violation: ``fill_null(0.0)`` should eliminate
    all null entries before this check is reached.  If this error is ever raised
    it indicates a logic defect in the cleaning step.

    Args:
        column: Name of the column that still contains null values.

    Examples:
        >>> raise NullsAfterCleaningError("A")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        basanos.exceptions.NullsAfterCleaningError: Column 'A' still contains null values...
    """

    def __init__(self, column: str) -> None:
        """Initialize with the name of the column that contains unexpected nulls."""
        super().__init__(
            f"Column '{column}' still contains null values after fill_null(0.0). "
            "This should never happen and indicates a logic defect in the cleaning step."
        )
        self.column = column


class NonFiniteAfterCleaningError(BasanosError, ValueError):
    """Raised when a profit column still contains non-finite values after the cleaning pass.

    This is an internal invariant violation: replacing all non-finite entries
    with ``0.0`` via ``otherwise(0.0)`` should guarantee finite values before
    this check is reached.  If this error is ever raised it indicates a logic
    defect in the cleaning step.

    Args:
        column: Name of the column that still contains non-finite values.

    Examples:
        >>> raise NonFiniteAfterCleaningError("A")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        basanos.exceptions.NonFiniteAfterCleaningError: Column 'A' still contains non-finite values...
    """

    def __init__(self, column: str) -> None:
        """Initialize with the name of the column that contains unexpected non-finite values."""
        super().__init__(
            f"Column '{column}' still contains non-finite values (NaN/Inf) after "
            "replacing all non-finite entries with 0.0. "
            "This should never happen and indicates a logic defect in the cleaning step."
        )
        self.column = column


class IllConditionedMatrixWarning(UserWarning):
    """Issued when a matrix has a condition number that exceeds a configured threshold.

    A high condition number indicates the matrix is nearly singular, and
    linear-algebra operations on it may produce numerically unreliable results.

    Examples:
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as w:
        ...     warnings.simplefilter("always")
        ...     warnings.warn("condition number 1e13", IllConditionedMatrixWarning)
        ...     assert len(w) == 1
    """


class MonotonicPricesError(BasanosError, ValueError):
    """Raised when an asset's price series is strictly monotonic.

    A monotonic series (all non-decreasing or all non-increasing) has no
    variance in its return sign, indicating malformed or synthetic data.

    Args:
        asset: Name of the offending asset column.

    Examples:
        >>> raise MonotonicPricesError("A")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        basanos.exceptions.MonotonicPricesError: Asset 'A' has monotonic prices...
    """

    def __init__(self, asset: str) -> None:
        """Initialize with the name of the asset that has monotonic prices."""
        super().__init__(
            f"Asset '{asset}' has monotonic prices "
            "(all non-decreasing or all non-increasing), indicating malformed or synthetic data."
        )
        self.asset = asset


class FactorLoadingsDimensionError(BasanosError, ValueError):
    """Raised when ``factor_loadings`` is not a 2-D array.

    Args:
        ndim: The actual number of dimensions of the array that was supplied.

    Examples:
        >>> raise FactorLoadingsDimensionError(1)
        Traceback (most recent call last):
            ...
        basanos.exceptions.FactorLoadingsDimensionError: factor_loadings must be 2-D, got ndim=1.
    """

    def __init__(self, ndim: int) -> None:
        """Initialize with the offending number of dimensions."""
        super().__init__(f"factor_loadings must be 2-D, got ndim={ndim}.")
        self.ndim = ndim


class FactorCovarianceShapeError(BasanosError, ValueError):
    """Raised when ``factor_covariance`` shape does not match the number of factors.

    Args:
        expected_k: Expected side length (number of factors *k*).
        got: Actual shape of the supplied ``factor_covariance`` array.

    Examples:
        >>> raise FactorCovarianceShapeError(2, (3, 3))
        Traceback (most recent call last):
            ...
        basanos.exceptions.FactorCovarianceShapeError: factor_covariance must have shape (2, 2) ...
    """

    def __init__(self, expected_k: int, got: tuple[int, ...]) -> None:
        """Initialize with the expected side length and the actual shape."""
        super().__init__(
            f"factor_covariance must have shape ({expected_k}, {expected_k}) to match "
            f"factor_loadings columns, got {got}."
        )
        self.expected_k = expected_k
        self.got = got


class IdiosyncraticVarShapeError(BasanosError, ValueError):
    """Raised when ``idiosyncratic_var`` length does not match the number of assets.

    Args:
        expected_n: Expected length (number of assets *n*).
        got: Actual shape of the supplied ``idiosyncratic_var`` array.

    Examples:
        >>> raise IdiosyncraticVarShapeError(4, (5,))
        Traceback (most recent call last):
            ...
        basanos.exceptions.IdiosyncraticVarShapeError: idiosyncratic_var must have shape (4,) ...
    """

    def __init__(self, expected_n: int, got: tuple[int, ...]) -> None:
        """Initialize with the expected length and the actual shape."""
        super().__init__(f"idiosyncratic_var must have shape ({expected_n},) to match factor_loadings rows, got {got}.")
        self.expected_n = expected_n
        self.got = got


class NonPositiveIdiosyncraticVarError(BasanosError, ValueError):
    """Raised when ``idiosyncratic_var`` contains zero or negative entries.

    All idiosyncratic variances must be strictly positive to ensure the
    factor model covariance matrix is positive definite.

    Examples:
        >>> raise NonPositiveIdiosyncraticVarError()
        Traceback (most recent call last):
            ...
        basanos.exceptions.NonPositiveIdiosyncraticVarError: All entries of idiosyncratic_var must be strictly positive.
    """

    def __init__(self) -> None:
        """Initialize the error."""
        super().__init__("All entries of idiosyncratic_var must be strictly positive.")


class ReturnMatrixDimensionError(BasanosError, ValueError):
    """Raised when the return matrix passed to ``FactorModel.from_returns`` is not 2-D.

    Args:
        ndim: The actual number of dimensions of the array that was supplied.

    Examples:
        >>> raise ReturnMatrixDimensionError(1)
        Traceback (most recent call last):
            ...
        basanos.exceptions.ReturnMatrixDimensionError: Return matrix must be 2-D, got ndim=1.
    """

    def __init__(self, ndim: int) -> None:
        """Initialize with the offending number of dimensions."""
        super().__init__(f"Return matrix must be 2-D, got ndim={ndim}.")
        self.ndim = ndim


class FactorCountError(BasanosError, ValueError):
    """Raised when the requested number of factors *k* is out of the valid range.

    Args:
        k: The requested number of factors.
        max_k: The maximum valid number of factors (``min(T, n)``).

    Examples:
        >>> raise FactorCountError(0, 5)
        Traceback (most recent call last):
            ...
        basanos.exceptions.FactorCountError: k must satisfy 1 <= k <= min(T, n) = 5, got k=0.
    """

    def __init__(self, k: int, max_k: int) -> None:
        """Initialize with the requested and maximum factor counts."""
        super().__init__(f"k must satisfy 1 <= k <= min(T, n) = {max_k}, got k={k}.")
        self.k = k
        self.max_k = max_k
