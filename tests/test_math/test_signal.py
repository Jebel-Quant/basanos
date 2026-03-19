"""Tests for basanos.math._signal (shrink2id, vol_adj).

shrink2id mixes a covariance/correlation matrix linearly towards the identity
matrix and is tested for boundary lambdas (0 and 1), midpoints, default
behaviour, and shape preservation.

vol_adj computes exponentially-weighted volatility-adjusted log-returns and
is tested for output shape, the mandatory leading null, finiteness of
subsequent values, ±clip enforcement, and the constant-price degenerate case.

factor_extract extracts k latent factors from a returns matrix R via truncated
SVD.  factor_project projects R onto the factor space to produce W = C^T @ R.
Both are tested for output shape, projection accuracy, and degenerate inputs.
"""

import math

import numpy as np
import polars as pl
import pytest

from basanos.math._signal import factor_extract, factor_project, shrink2id, vol_adj

# ─── shrink2id ────────────────────────────────────────────────────────────────


def test_shrink2id_behavior():
    """shrink2id mixes a matrix with identity across lambda values."""
    mat = np.array([[2.0, 1.0], [1.0, 3.0]])

    # lamb=1.0 -> original
    np.testing.assert_allclose(shrink2id(mat, lamb=1.0), mat)

    # lamb=0.0 -> identity
    np.testing.assert_allclose(shrink2id(mat, lamb=0.0), np.eye(2))

    # lamb=0.5 -> halfway between M and I
    np.testing.assert_allclose(shrink2id(mat, lamb=0.5), 0.5 * mat + 0.5 * np.eye(2))


def test_shrink2id_default_lamb_is_one():
    """Default lamb=1.0 must return the original matrix unchanged."""
    mat = np.array([[3.0, 0.5], [0.5, 2.0]])
    np.testing.assert_array_equal(shrink2id(mat), mat)


def test_shrink2id_output_shape_preserved():
    """shrink2id must not change the shape of the input matrix for any lamb.

    This guards against implementations that, e.g., rebuild from eigenvalues
    and inadvertently drop dimensions.
    """
    mat = np.eye(4)
    assert shrink2id(mat, lamb=0.3).shape == mat.shape


# ─── vol_adj ─────────────────────────────────────────────────────────────────


@pytest.fixture
def price_series() -> pl.DataFrame:
    """Ten-row price series with small realistic fluctuations and no NaN values.

    Prices oscillate gently around a mild uptrend so that log-returns and the
    EWM volatility estimator both receive non-trivial, non-degenerate inputs.
    """
    prices = [1.0, 1.01, 1.03, 1.02, 1.05, 1.04, 1.06, 1.08, 1.07, 1.10]
    return pl.DataFrame({"p": prices})


def test_vol_adj_output_shape(price_series):
    """vol_adj must return a series of the same length as input."""
    result = price_series.select(vol_adj(pl.col("p"), vola=2, clip=3.0))
    assert result.shape == price_series.shape


def test_vol_adj_first_row_is_null(price_series):
    """First value must be null because log().diff() produces null at index 0."""
    result = price_series.select(vol_adj(pl.col("p"), vola=2, clip=3.0))
    assert result["p"][0] is None


def test_vol_adj_subsequent_rows_are_finite(price_series):
    """All non-null values (index >= 1) must be finite floats."""
    result = price_series.select(vol_adj(pl.col("p"), vola=2, clip=3.0))
    vals = result["p"].drop_nulls().to_list()
    assert len(vals) > 0
    assert all(math.isfinite(v) for v in vals)


def test_vol_adj_clipping_respected():
    """vol_adj must clamp all non-null, non-NaN outputs to the interval [−clip, +clip].

    Uses a large price spike at the end (×100 jump) to force the standardised
    log-return far beyond the clip bound, confirming that the clamp is applied
    after volatility normalisation.
    """
    clip = 1.0
    # Prices with prior volatility (so EWM std is non-zero) then a huge jump.
    prices = [1.0, 1.05, 0.95, 1.05, 0.95, 1.05, 100.0]
    df = pl.DataFrame({"p": prices})
    result = df.select(vol_adj(pl.col("p"), vola=2, clip=clip))
    # Drop nulls and NaNs (NaN arises from 0/0 which clip cannot bound)
    vals = [v for v in result["p"].to_list() if v is not None and not math.isnan(v)]
    assert len(vals) > 0
    assert all(v <= clip + 1e-12 for v in vals)
    assert all(v >= -clip - 1e-12 for v in vals)


def test_vol_adj_constant_price_returns_null_or_nan():
    """Constant price series has zero log-returns; vol is zero → standardized returns are null/NaN."""
    df = pl.DataFrame({"p": [5.0] * 8})
    result = df.select(vol_adj(pl.col("p"), vola=2, clip=3.0))
    # All values should be either null or NaN (division by zero vol)
    for v in result["p"].to_list():
        assert v is None or (isinstance(v, float) and math.isnan(v))


# ─── factor_extract ───────────────────────────────────────────────────────────


def test_factor_extract_shape():
    """factor_extract must return a factor matrix of shape (n, k)."""
    rng = np.random.default_rng(0)
    ret_mat = rng.normal(size=(20, 5))
    factor_mat = factor_extract(ret_mat, k=3)
    assert factor_mat.shape == (20, 3)


def test_factor_extract_shape_k_equals_m():
    """factor_extract with k=m must return a factor matrix of shape (n, m)."""
    rng = np.random.default_rng(1)
    ret_mat = rng.normal(size=(10, 4))
    factor_mat = factor_extract(ret_mat, k=4)
    assert factor_mat.shape == (10, 4)


def test_factor_extract_columns_orthonormal():
    """Columns of the factor matrix must be orthonormal (F^T F ≈ I_k) up to floating-point error."""
    rng = np.random.default_rng(2)
    ret_mat = rng.normal(size=(30, 6))
    k = 4
    factor_mat = factor_extract(ret_mat, k=k)
    np.testing.assert_allclose(factor_mat.T @ factor_mat, np.eye(k), atol=1e-10)


def test_factor_extract_k1_returns_single_column():
    """factor_extract with k=1 must return a column vector of shape (n, 1)."""
    rng = np.random.default_rng(3)
    ret_mat = rng.normal(size=(15, 3))
    factor_mat = factor_extract(ret_mat, k=1)
    assert factor_mat.shape == (15, 1)


# ─── factor_project ───────────────────────────────────────────────────────────


def test_factor_project_shape():
    """factor_project must return a projection of shape (k, m)."""
    rng = np.random.default_rng(4)
    ret_mat = rng.normal(size=(20, 5))
    factor_mat = factor_extract(ret_mat, k=3)
    projection = factor_project(factor_mat, ret_mat)
    assert projection.shape == (3, 5)


def test_factor_project_equals_ft_times_r():
    """factor_project(factor_mat, ret_mat) must equal factor_mat.T @ ret_mat exactly."""
    rng = np.random.default_rng(5)
    ret_mat = rng.normal(size=(20, 5))
    factor_mat = factor_extract(ret_mat, k=3)
    projection = factor_project(factor_mat, ret_mat)
    np.testing.assert_array_equal(projection, factor_mat.T @ ret_mat)


def test_factor_project_shape_k_equals_m():
    """With k=m the projection must have shape (m, m)."""
    rng = np.random.default_rng(6)
    m = 4
    ret_mat = rng.normal(size=(10, m))
    factor_mat = factor_extract(ret_mat, k=m)
    projection = factor_project(factor_mat, ret_mat)
    assert projection.shape == (m, m)


def test_factor_project_k1_shape():
    """factor_project with k=1 must return a projection of shape (1, m)."""
    rng = np.random.default_rng(7)
    ret_mat = rng.normal(size=(15, 3))
    factor_mat = factor_extract(ret_mat, k=1)
    projection = factor_project(factor_mat, ret_mat)
    assert projection.shape == (1, 3)
