"""Linear algebra helpers used by the Basanos optimizer.

This private module provides a small set of NumPy-based utilities for
working with symmetric (correlation-like) matrices in a robust way:
- valid(matrix): mask out rows/cols with non-finite diagonal entries and
  return the corresponding sub-matrix.
- is_positive_definite(matrix): explicitly test whether a matrix is
  positive-definite using Cholesky decomposition.
- inv_a_norm(vector, matrix=None, cond_threshold=...): compute the inverse
  A-norm of a vector with respect to a (possibly masked) positive-definite
  matrix; defaults to the Euclidean norm when no matrix is given. Emits
  IllConditionedMatrixWarning when the condition number exceeds the threshold.
- solve(matrix, rhs, cond_threshold=...): solve a linear system on the valid
  subset indicated by finite diagonal entries, returning NaNs for invalid
  positions. Emits IllConditionedMatrixWarning when the condition number
  exceeds the threshold.

Both solve and inv_a_norm use Cholesky decomposition as a numerically stable
first attempt; they fall back to standard LU-based solving for matrices that
are not positive-definite.

These routines are intentionally lightweight and raise domain-specific
exceptions to guard against shape mismatches and numerical errors.
They are internal implementation details and may change without notice.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from basanos.exceptions import (
    DimensionMismatchError,
    IllConditionedMatrixWarning,
    NonSquareMatrixError,
    SingularMatrixError,
)

_DEFAULT_COND_THRESHOLD: float = 1e12
"""Default condition-number threshold above which a warning is emitted."""


def valid(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate a square matrix and return mask and valid sub-matrix.

    Checks that the input is square and that diagonal entries are finite.
    Returns a boolean mask for valid diagonal entries and the sub-matrix
    consisting of valid rows/columns.

    Args:
        matrix (np.ndarray): Input square matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - mask: Boolean array indicating which diagonal elements are finite.
            - submatrix: Matrix containing only rows/columns with finite diagonal.

    Raises:
        NonSquareMatrixError: If the input matrix is not square.

    Examples:
        >>> import numpy as np
        >>> mask, sub = valid(np.eye(2))
        >>> mask.tolist()
        [True, True]
        >>> sub.shape
        (2, 2)
    """
    # make sure matrix  is quadratic
    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError(matrix.shape[0], matrix.shape[1])

    v = np.isfinite(np.diag(matrix))
    return v, matrix[:, v][v]


def is_positive_definite(matrix: np.ndarray) -> bool:
    """Return True if *matrix* is symmetric positive-definite, False otherwise.

    The check is performed via an attempted Cholesky decomposition — the most
    numerically reliable way to test positive-definiteness for symmetric
    matrices.  A matrix is positive-definite if and only if its Cholesky
    decomposition exists and all diagonal entries of the Cholesky factor are
    strictly positive.

    This function is intentionally side-effect-free: it raises no exceptions
    and emits no warnings.  It is suitable for use as a guard before passing a
    shrunk correlation matrix to a linear solver.

    Args:
        matrix (np.ndarray): Square matrix to test.

    Returns:
        bool: ``True`` if the matrix is positive-definite, ``False`` otherwise.

    Examples:
        >>> import numpy as np
        >>> is_positive_definite(np.eye(3))
        True
        >>> is_positive_definite(np.array([[1.0, 2.0], [2.0, 1.0]]))
        False
        >>> is_positive_definite(np.array([[1.0, 0.5], [0.5, 1.0]]))
        True
    """
    try:
        cho_factor(matrix, check_finite=False)
    except np.linalg.LinAlgError:
        return False
    else:
        return True


def _check_and_warn_condition(matrix: np.ndarray, threshold: float) -> None:
    """Emit IllConditionedMatrixWarning when the condition number exceeds threshold.

    Args:
        matrix (np.ndarray): Square matrix whose condition number is checked.
        threshold (float): Upper bound before a warning is issued.
    """
    cond = float(np.linalg.cond(matrix))
    if cond > threshold:
        warnings.warn(
            f"Matrix condition number {cond:.3e} exceeds threshold {threshold:.3e}; "
            "results may be numerically unreliable.",
            IllConditionedMatrixWarning,
            stacklevel=3,
        )


def _cholesky_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve *matrix* x = *rhs* using Cholesky, falling back to LU if needed.

    Cholesky decomposition is attempted first as it is numerically more stable
    for positive-definite matrices.  If the decomposition fails (the matrix is
    not positive-definite), a standard LU-based solve is used instead.

    Args:
        matrix (np.ndarray): Square coefficient matrix.
        rhs (np.ndarray): Right-hand side vector.

    Returns:
        np.ndarray: Solution vector.

    Raises:
        np.linalg.LinAlgError: If both Cholesky and LU-based solves fail.
    """
    try:
        c, lower = cho_factor(matrix, check_finite=False)
        return cho_solve((c, lower), rhs, check_finite=False)
    except np.linalg.LinAlgError:
        # Matrix is not positive-definite; fall back to general LU solve.
        return np.linalg.solve(matrix, rhs)


def inv_a_norm(
    vector: np.ndarray,
    matrix: np.ndarray | None = None,
    cond_threshold: float = _DEFAULT_COND_THRESHOLD,
) -> float:
    """Compute inverse A-norm of a vector with an optional metric matrix.

    If ``matrix`` is None, compute the Euclidean norm of finite entries.
    Otherwise, validate the matrix is square and dimensionally compatible,
    then return sqrt(v^T A^{-1} v) on the valid subset.

    Cholesky decomposition is attempted first for numerical stability; the
    solver falls back to LU decomposition for non-positive-definite matrices.
    If the condition number of the valid sub-matrix exceeds *cond_threshold*,
    an :class:`~basanos.exceptions.IllConditionedMatrixWarning` is emitted.

    Args:
        vector (np.ndarray): Input vector.
        matrix (np.ndarray | None, optional): Positive-definite metric matrix.
        cond_threshold (float, optional): Condition-number threshold above which
            a warning is emitted. Defaults to ``1e12``.

    Returns:
        float: The computed norm. Returns np.nan if no valid entries exist.

    Raises:
        NonSquareMatrixError: If ``matrix`` is not square.
        DimensionMismatchError: If ``vector`` length does not match the matrix dimension.
        SingularMatrixError: If the valid sub-matrix is singular.

    Examples:
        >>> import numpy as np
        >>> inv_a_norm(np.array([3.0, 4.0]))
        5.0
    """
    if matrix is None:
        return float(np.linalg.norm(vector[np.isfinite(vector)], 2))

    # make sure matrix is quadratic
    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError(matrix.shape[0], matrix.shape[1])

    # make sure the vector has the right number of entries
    if vector.size != matrix.shape[0]:
        raise DimensionMismatchError(vector.size, matrix.shape[0])

    v, mat = valid(matrix)

    if v.any():
        _check_and_warn_condition(mat, cond_threshold)
        try:
            return float(np.sqrt(np.dot(vector[v], _cholesky_solve(mat, vector[v]))))
        except np.linalg.LinAlgError as exc:
            raise SingularMatrixError(str(exc)) from exc
    return float(np.nan)


def solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
    cond_threshold: float = _DEFAULT_COND_THRESHOLD,
) -> np.ndarray:
    """Solve Ax=b on the valid subset indicated by finite diagonal entries.

    Cholesky decomposition is attempted first for numerical stability; the
    solver falls back to LU decomposition for non-positive-definite matrices.
    If the condition number of the valid sub-matrix exceeds *cond_threshold*,
    an :class:`~basanos.exceptions.IllConditionedMatrixWarning` is emitted.

    Args:
        matrix (np.ndarray): Square coefficient matrix A.
        rhs (np.ndarray): Right-hand side vector b with matching size.
        cond_threshold (float, optional): Condition-number threshold above which
            a warning is emitted. Defaults to ``1e12``.

    Returns:
        np.ndarray: Solution vector with NaN for invalid positions.

    Raises:
        NonSquareMatrixError: If matrix is not square.
        DimensionMismatchError: If rhs size does not match the matrix dimension.
        SingularMatrixError: If the valid sub-matrix is singular.

    Examples:
        >>> import numpy as np
        >>> solve(np.eye(2), np.array([1.0, 2.0])).tolist()
        [1.0, 2.0]
    """
    # make sure matrix is quadratic
    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError(matrix.shape[0], matrix.shape[1])

    # make sure the vector rhs has the right number of entries
    if rhs.size != matrix.shape[0]:
        raise DimensionMismatchError(rhs.size, matrix.shape[0])

    x = np.nan * np.ones(rhs.size)
    v, mat = valid(matrix)

    if v.any():
        _check_and_warn_condition(mat, cond_threshold)
        try:
            x[v] = _cholesky_solve(mat, rhs[v])
        except np.linalg.LinAlgError as exc:
            raise SingularMatrixError(str(exc)) from exc

    return x
