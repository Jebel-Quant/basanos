"""Linear algebra helpers used by the Basanos optimizer.

This private module provides a small set of NumPy-based utilities for
working with symmetric (correlation-like) matrices in a robust way:
- valid(matrix): mask out rows/cols with non-finite diagonal entries and
  return the corresponding sub-matrix.
- inv_a_norm(vector, matrix=None): compute the inverse A-norm of a vector
  with respect to a (possibly masked) positive-definite matrix; defaults
  to the Euclidean norm when no matrix is given.
- solve(matrix, rhs): solve a linear system on the valid subset indicated
  by finite diagonal entries, returning NaNs for invalid positions.

These routines are intentionally lightweight and have explicit input
assertions to guard against shape mismatches. They are internal
implementation details and may change without notice.
"""

from __future__ import annotations

import numpy as np


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
        AssertionError: If the input matrix is not square.
    """
    # make sure matrix  is quadratic
    if matrix.shape[0] != matrix.shape[1]:
        raise AssertionError

    v = np.isfinite(np.diag(matrix))
    return v, matrix[:, v][v]


def inv_a_norm(vector: np.ndarray, matrix: np.ndarray | None = None) -> float:
    """Compute inverse A-norm of a vector with an optional metric matrix.

    If ``matrix`` is None, compute the Euclidean norm of finite entries.
    Otherwise, validate the matrix is square and dimensionally compatible,
    then return sqrt(v^T A^{-1} v) on the valid subset.

    Args:
        vector (np.ndarray): Input vector.
        matrix (np.ndarray | None, optional): Positive-definite metric matrix.

    Returns:
        float: The computed norm. Returns np.nan if no valid entries exist.

    Raises:
        AssertionError: If ``matrix`` is not square or dimensions mismatch.
    """
    if matrix is None:
        return float(np.linalg.norm(vector[np.isfinite(vector)], 2))

    # make sure matrix is quadratic
    if matrix.shape[0] != matrix.shape[1]:
        raise AssertionError

    # make sure the vector has the right number of entries
    if vector.size != matrix.shape[0]:
        raise AssertionError

    v, mat = valid(matrix)

    if v.any():
        return float(np.sqrt(np.dot(vector[v], np.linalg.solve(mat, vector[v]))))
    return float(np.nan)


def solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve Ax=b on the valid subset indicated by finite diagonal entries.

    Args:
        matrix (np.ndarray): Square coefficient matrix A.
        rhs (np.ndarray): Right-hand side vector b with matching size.

    Returns:
        np.ndarray: Solution vector with NaN for invalid positions.

    Raises:
        AssertionError: If matrix is not square or rhs size mismatches.
    """
    # make sure matrix is quadratic
    if matrix.shape[0] != matrix.shape[1]:
        raise AssertionError

    # make sure the vector rhs has the right number of entries
    if rhs.size != matrix.shape[0]:
        raise AssertionError

    x = np.nan * np.ones(rhs.size)
    v, mat = valid(matrix)

    if v.any():
        x[v] = np.linalg.solve(mat, rhs[v])

    return x
