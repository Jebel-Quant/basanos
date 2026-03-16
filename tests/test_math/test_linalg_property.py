"""Property-based tests for basanos.math._linalg.

Uses Hypothesis to systematically explore edge cases for the three public
helpers in _linalg: ``valid``, ``inv_a_norm``, and ``solve``.

Properties under test
---------------------
valid
  - The returned mask always has the same length as the matrix dimension.
  - A matrix whose diagonal is entirely finite yields an all-True mask.
  - The returned sub-matrix has side length equal to the number of True entries.

inv_a_norm (matrix=None)
  - Result equals ``np.linalg.norm(v[np.isfinite(v)], 2)`` for every input.
  - Result is always non-negative.

inv_a_norm (matrix=identity)
  - With an identity matrix the A-norm collapses to the Euclidean norm.

solve (matrix=identity)
  - The solution of I·x = b equals b.

solve (diagonal matrix)
  - The solution satisfies matrix @ x ≈ rhs at valid (non-NaN) positions.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_st

from basanos.math._linalg import inv_a_norm, solve, valid

# ─── Shared strategies ────────────────────────────────────────────────────────

_SIZES = st.integers(min_value=1, max_value=8)
_FINITE_FLOAT = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)
_MAYBE_NAN_FLOAT = st.floats(allow_nan=True, allow_infinity=False)


def _square_matrix(n: int) -> st.SearchStrategy[np.ndarray]:
    """Strategy for an (n, n) float64 matrix with possibly-NaN entries."""
    return np_st.arrays(dtype=np.float64, shape=(n, n), elements=_MAYBE_NAN_FLOAT)


def _finite_square_matrix(n: int) -> st.SearchStrategy[np.ndarray]:
    """Strategy for an (n, n) finite float64 matrix."""
    return np_st.arrays(dtype=np.float64, shape=(n, n), elements=_FINITE_FLOAT)


def _finite_vector(n: int) -> st.SearchStrategy[np.ndarray]:
    """Strategy for a length-n finite float64 vector."""
    return np_st.arrays(dtype=np.float64, shape=(n,), elements=_FINITE_FLOAT)


# ─── valid() ─────────────────────────────────────────────────────────────────


@given(n=_SIZES, data=st.data())
@settings(max_examples=200)
def test_valid_mask_length_equals_matrix_size(n: int, data: st.DataObject) -> None:
    """valid() must return a mask whose length equals the matrix dimension."""
    matrix = data.draw(_square_matrix(n))
    mask, _ = valid(matrix)
    assert len(mask) == n


@given(n=_SIZES, data=st.data())
@settings(max_examples=200)
def test_valid_finite_diagonal_gives_all_true_mask(n: int, data: st.DataObject) -> None:
    """A matrix whose diagonal is entirely finite must yield an all-True mask."""
    matrix = data.draw(_finite_square_matrix(n))
    mask, submatrix = valid(matrix)
    assert mask.all()
    assert submatrix.shape == (n, n)


@given(n=_SIZES, data=st.data())
@settings(max_examples=200)
def test_valid_submatrix_shape_matches_true_count(n: int, data: st.DataObject) -> None:
    """The sub-matrix side length must equal the number of True entries in the mask."""
    matrix = data.draw(_square_matrix(n))
    mask, submatrix = valid(matrix)
    k = int(mask.sum())
    assert submatrix.shape == (k, k)


# ─── inv_a_norm (no matrix) ───────────────────────────────────────────────────


@given(
    vector=np_st.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=20),
        elements=st.floats(allow_nan=True, allow_infinity=False),
    )
)
@settings(max_examples=300)
def test_inv_a_norm_no_matrix_equals_euclidean_norm(vector: np.ndarray) -> None:
    """inv_a_norm(v) must equal the Euclidean norm of the finite entries in v."""
    result = inv_a_norm(vector)
    expected = float(np.linalg.norm(vector[np.isfinite(vector)], 2))
    assert result == pytest.approx(expected, abs=1e-10)


@given(
    vector=np_st.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=20),
        elements=_FINITE_FLOAT,
    )
)
@settings(max_examples=300)
def test_inv_a_norm_no_matrix_is_non_negative(vector: np.ndarray) -> None:
    """The Euclidean norm of any finite vector is always >= 0."""
    result = inv_a_norm(vector)
    assert result >= 0.0


# ─── inv_a_norm (identity matrix) ────────────────────────────────────────────


@given(n=_SIZES, data=st.data())
@settings(max_examples=200)
def test_inv_a_norm_with_identity_equals_euclidean_norm(n: int, data: st.DataObject) -> None:
    """With an identity matrix, the inverse A-norm collapses to the Euclidean norm."""
    vector = data.draw(_finite_vector(n))
    result = inv_a_norm(vector=vector, matrix=np.eye(n))
    expected = float(np.linalg.norm(vector, 2))
    assert result == pytest.approx(expected, rel=1e-9, abs=1e-12)


# ─── solve (identity) ────────────────────────────────────────────────────────


@given(n=_SIZES, data=st.data())
@settings(max_examples=200)
def test_solve_identity_returns_rhs(n: int, data: st.DataObject) -> None:
    """Solving I·x = b must return b exactly."""
    rhs = data.draw(_finite_vector(n))
    result = solve(matrix=np.eye(n), rhs=rhs)
    np.testing.assert_allclose(result, rhs, rtol=1e-10, atol=1e-12)


# ─── solve (positive diagonal matrix) ────────────────────────────────────────


@given(n=_SIZES, data=st.data())
@settings(max_examples=200)
def test_solve_diagonal_satisfies_linear_system(n: int, data: st.DataObject) -> None:
    """For a positive-definite diagonal matrix, matrix @ solve(matrix, rhs) == rhs."""
    # Draw strictly positive diagonal entries so the matrix is non-singular.
    diag_vals = data.draw(
        np_st.arrays(
            dtype=np.float64,
            shape=(n,),
            elements=st.floats(min_value=0.1, max_value=1e3, allow_nan=False, allow_infinity=False),
        )
    )
    matrix = np.diag(diag_vals)
    rhs = data.draw(_finite_vector(n))

    x = solve(matrix=matrix, rhs=rhs)

    # All entries should be finite and the system should be satisfied.
    assert np.all(np.isfinite(x))
    np.testing.assert_allclose(matrix @ x, rhs, rtol=1e-9, atol=1e-9)
