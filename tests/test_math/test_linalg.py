"""Tests for the linear algebra module of TinyCTA.

This module contains tests for the linear algebra functions in the TinyCTA package.
It tests various matrix operations including validation, norm calculations, and
solving linear systems with different input scenarios including edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from basanos.exceptions import (
    DimensionMismatchError,
    NonSquareMatrixError,
    SingularMatrixError,
)
from basanos.math._linalg import inv_a_norm, solve, valid


def test_non_quadratic() -> None:
    """Test that functions properly reject non-square matrices.

    This test verifies that all linear algebra functions that require square matrices
    (valid, a_norm, inv_a_norm, solve) correctly raise NonSquareMatrixError when
    provided with non-square matrices, and that the error message contains the shape.

    Each function is tested with a non-square matrix input to ensure proper validation.
    """
    with pytest.raises(NonSquareMatrixError, match=r"shape \(1, 2\)"):
        valid(np.array([[2.0, 1.0]]))

    with pytest.raises(NonSquareMatrixError, match=r"shape \(1, 2\)"):
        inv_a_norm(vector=np.array([2.0, 1.0]), matrix=np.array([[2.0, 1.0]]))

    with pytest.raises(NonSquareMatrixError, match=r"shape \(1, 2\)"):
        solve(matrix=np.array([[2.0, 1.0]]), rhs=np.array([2.0, 1.0]))


def test_mismatch() -> None:
    """Test that functions properly reject mismatched dimensions.

    This test verifies that linear algebra functions correctly raise
    DimensionMismatchError when the dimensions of the matrix and vector don't match
    (e.g., when a 1x1 matrix is used with a 2-element vector), and that the message
    contains the offending sizes.

    Each function is tested with mismatched dimensions to ensure proper validation.
    """
    with pytest.raises(DimensionMismatchError, match=r"length 2.*dimension 1"):
        inv_a_norm(vector=np.array([1.0, 2.0]), matrix=np.array([[1.0]]))

    with pytest.raises(DimensionMismatchError, match=r"length 2.*dimension 1"):
        solve(matrix=np.array([[1.0]]), rhs=np.array([1.0, 2.0]))


def test_valid() -> None:
    """Test the valid function with a partially NaN matrix.

    This test verifies that the valid function correctly identifies valid rows/columns
    in a matrix containing NaN values and returns the appropriate submatrix.

    The test creates a 2x2 matrix with NaN values in specific positions and checks
    that the valid function correctly identifies the valid elements and extracts
    the appropriate submatrix.
    """
    a = np.array([[1.0, np.nan], [np.nan, np.nan]])
    val, submatrix = valid(a)

    np.testing.assert_array_equal(val, np.array([True, False]))
    np.testing.assert_array_equal(submatrix, np.array([[1.0]]))


def test_valid_eye() -> None:
    """Test the valid function with an identity matrix.

    This test verifies that the valid function correctly processes an identity matrix
    where all elements are valid (no NaN values).

    The test creates a 2x2 identity matrix and checks that the valid function
    correctly identifies all elements as valid and returns the original matrix.
    """
    a = np.eye(2)
    val, submatrix = valid(a)

    np.testing.assert_array_equal(val, np.array([True, True]))
    np.testing.assert_array_equal(submatrix, np.eye(2))


def test_inv_a_norm() -> None:
    """Test the inv_a_norm function with a matrix parameter.

    This test verifies that the inv_a_norm function correctly calculates the inverse
    matrix norm of a vector with respect to a given matrix.

    The test creates a vector [3.0, 4.0] and a matrix 0.5*I, then checks that
    inv_a_norm returns sqrt(2)*5.0, which is the correct inverse matrix norm.
    This is equivalent to the matrix norm with respect to the inverse matrix.
    """
    v = np.array([3.0, 4.0])
    a = 0.5 * np.eye(2)
    assert inv_a_norm(vector=v, matrix=a) == pytest.approx(np.sqrt(2) * 5.0)


def test_inv_a_norm_without_matrix() -> None:
    """Test the inv_a_norm function without providing a matrix parameter.

    This test verifies that the inv_a_norm function correctly calculates the 2-norm
    of a vector when no matrix is provided, which is the same behavior as a_norm.

    The test creates a vector [3.0, 4.0] and checks that inv_a_norm returns 5.0,
    which is the Euclidean norm (sqrt(3^2 + 4^2)).
    """
    v = np.array([3.0, 4.0])
    assert inv_a_norm(vector=v) == pytest.approx(5.0)


def test_inv_a_norm_all_nan() -> None:
    """Test the inv_a_norm function with a matrix containing all NaN values.

    This test verifies that the inv_a_norm function correctly handles the case where
    the matrix contains all NaN values, returning NaN as the result.

    The test creates a vector [3.0, 4.0] and a matrix of NaNs, then checks that
    inv_a_norm returns NaN.
    """
    v = np.array([3.0, 4.0])
    a = np.nan * np.eye(2)
    assert np.isnan(inv_a_norm(vector=v, matrix=a))


def test_solve() -> None:
    """Test the solve function for solving a linear system.

    This test verifies that the solve function correctly solves the linear system
    matrix * x = rhs for x, where matrix is a square matrix and rhs is a vector.

    The test creates a right-hand side vector [3.0, 4.0] and a matrix 0.5*I,
    then checks that the solution x satisfies the original equation matrix * x = rhs.
    For this specific case, the expected solution is [6.0, 8.0].
    """
    rhs = np.array([3.0, 4.0])
    matrix = 0.5 * np.eye(2)
    x = solve(matrix=matrix, rhs=rhs)

    np.testing.assert_array_equal(matrix @ x, rhs)


def test_singular_matrix_solve() -> None:
    """Test that solve raises SingularMatrixError for a singular matrix.

    This test verifies that the solve function raises a SingularMatrixError with
    a descriptive message when the matrix is singular (non-invertible).
    """
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    rhs = np.array([1.0, 2.0])
    with pytest.raises(SingularMatrixError, match="singular"):
        solve(matrix=singular, rhs=rhs)


def test_singular_matrix_inv_a_norm() -> None:
    """Test that inv_a_norm raises SingularMatrixError for a singular matrix.

    This test verifies that the inv_a_norm function raises a SingularMatrixError
    with a descriptive message when the metric matrix is singular.
    """
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    v = np.array([1.0, 2.0])
    with pytest.raises(SingularMatrixError, match="singular"):
        inv_a_norm(vector=v, matrix=singular)


def test_exception_hierarchy() -> None:
    """Test that custom exceptions inherit from BasanosError and ValueError.

    Verifies the intended inheritance hierarchy so callers can catch either
    the full family (BasanosError) or the standard ValueError family.
    """
    from basanos.exceptions import BasanosError

    assert issubclass(NonSquareMatrixError, BasanosError)
    assert issubclass(NonSquareMatrixError, ValueError)
    assert issubclass(DimensionMismatchError, BasanosError)
    assert issubclass(DimensionMismatchError, ValueError)
    assert issubclass(SingularMatrixError, BasanosError)
    assert issubclass(SingularMatrixError, ValueError)


def test_non_square_matrix_error_attributes() -> None:
    """Test that NonSquareMatrixError stores the offending dimensions."""
    exc = NonSquareMatrixError(3, 2)
    assert exc.rows == 3
    assert exc.cols == 2
    assert "3" in str(exc)
    assert "2" in str(exc)


def test_dimension_mismatch_error_attributes() -> None:
    """Test that DimensionMismatchError stores the offending sizes."""
    exc = DimensionMismatchError(5, 3)
    assert exc.vector_size == 5
    assert exc.matrix_size == 3
    assert "5" in str(exc)
    assert "3" in str(exc)


def test_insufficient_data_error() -> None:
    """Test InsufficientDataError with and without a detail message."""
    from basanos.exceptions import BasanosError, InsufficientDataError

    exc_default = InsufficientDataError()
    assert "Insufficient" in str(exc_default)
    assert issubclass(InsufficientDataError, BasanosError)
    assert issubclass(InsufficientDataError, ValueError)

    detail = "All diagonal entries are non-finite."
    exc_detail = InsufficientDataError(detail)
    assert str(exc_detail) == detail
