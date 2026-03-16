"""Property-based tests for basanos.math._signal.

Uses Hypothesis to systematically explore edge cases for ``shrink2id``
and ``vol_adj``.

Properties under test
---------------------
shrink2id
  - Output shape is always identical to the input shape.
  - ``lamb=0`` always returns the identity matrix.
  - ``lamb=1`` always returns the original matrix.
  - For any ``lamb`` in [0, 1] the result equals the linear interpolation
    ``lamb * M + (1 - lamb) * I``.

vol_adj
  - Output has the same number of rows as the input.
  - The first element of the output is always null (from ``log().diff()``).
  - All non-null, non-NaN outputs are within the symmetric clip bounds.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_st

from basanos.math._signal import shrink2id, vol_adj

# ─── Shared strategies ────────────────────────────────────────────────────────

_SIZES = st.integers(min_value=1, max_value=8)
_FINITE_FLOAT = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
_LAMB = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _square_matrix(n: int) -> st.SearchStrategy[np.ndarray]:
    """Strategy for an (n, n) finite float64 matrix."""
    return np_st.arrays(dtype=np.float64, shape=(n, n), elements=_FINITE_FLOAT)


# ─── shrink2id ────────────────────────────────────────────────────────────────


@given(n=_SIZES, lamb=_LAMB, data=st.data())
@settings(max_examples=300)
def test_shrink2id_shape_preserved(n: int, lamb: float, data: st.DataObject) -> None:
    """shrink2id must not alter the shape of the input matrix."""
    matrix = data.draw(_square_matrix(n))
    result = shrink2id(matrix, lamb=lamb)
    assert result.shape == (n, n)


@given(n=_SIZES, data=st.data())
@settings(max_examples=200)
def test_shrink2id_lamb_zero_gives_identity(n: int, data: st.DataObject) -> None:
    """shrink2id(M, 0) must return the identity matrix regardless of M."""
    matrix = data.draw(_square_matrix(n))
    result = shrink2id(matrix, lamb=0.0)
    np.testing.assert_allclose(result, np.eye(n), atol=1e-12)


@given(n=_SIZES, data=st.data())
@settings(max_examples=200)
def test_shrink2id_lamb_one_gives_original(n: int, data: st.DataObject) -> None:
    """shrink2id(M, 1) must return M exactly."""
    matrix = data.draw(_square_matrix(n))
    result = shrink2id(matrix, lamb=1.0)
    np.testing.assert_array_equal(result, matrix)


@given(n=_SIZES, lamb=_LAMB, data=st.data())
@settings(max_examples=300)
def test_shrink2id_is_linear_interpolation(n: int, lamb: float, data: st.DataObject) -> None:
    """shrink2id(M, t) must equal t·M + (1-t)·I for any t in [0, 1]."""
    matrix = data.draw(_square_matrix(n))
    result = shrink2id(matrix, lamb=lamb)
    expected = lamb * matrix + (1.0 - lamb) * np.eye(n)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


# ─── vol_adj ─────────────────────────────────────────────────────────────────


@given(
    prices=st.lists(
        st.floats(min_value=0.01, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=30,
    ),
    vola=st.integers(min_value=2, max_value=10),
    clip=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300)
def test_vol_adj_output_length_matches_input(prices: list[float], vola: int, clip: float) -> None:
    """vol_adj output must have the same number of rows as the input."""
    df = pl.DataFrame({"p": prices})
    result = df.select(vol_adj(pl.col("p"), vola=vola, clip=clip))
    assert result.height == len(prices)


@given(
    prices=st.lists(
        st.floats(min_value=0.01, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=30,
    ),
    vola=st.integers(min_value=2, max_value=10),
    clip=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300)
def test_vol_adj_first_value_is_null(prices: list[float], vola: int, clip: float) -> None:
    """The first output value must always be null (log().diff() produces null at index 0)."""
    df = pl.DataFrame({"p": prices})
    result = df.select(vol_adj(pl.col("p"), vola=vola, clip=clip))
    assert result["p"][0] is None


@given(
    prices=st.lists(
        st.floats(min_value=0.01, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=30,
    ),
    vola=st.integers(min_value=2, max_value=10),
    clip=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300)
def test_vol_adj_non_null_values_within_clip_bounds(prices: list[float], vola: int, clip: float) -> None:
    """All non-null, non-NaN values from vol_adj must lie within [-clip, +clip]."""
    df = pl.DataFrame({"p": prices})
    result = df.select(vol_adj(pl.col("p"), vola=vola, clip=clip))
    vals = [v for v in result["p"].to_list() if v is not None and not math.isnan(v)]
    tolerance = 1e-10
    assert all(-clip - tolerance <= v <= clip + tolerance for v in vals), (
        f"Some values exceeded clip={clip}: {[v for v in vals if abs(v) > clip + tolerance]}"
    )
