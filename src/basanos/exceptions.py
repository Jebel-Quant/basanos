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
