"""Tests for basanos.math._signal (shrink2id, vol_adj, pca_cov).

shrink2id mixes a covariance/correlation matrix linearly towards the identity
matrix and is tested for boundary lambdas (0 and 1), midpoints, default
behaviour, and shape preservation.

vol_adj computes exponentially-weighted volatility-adjusted log-returns and
is tested for output shape, the mandatory leading null, finiteness of
subsequent values, ±clip enforcement, and the constant-price degenerate case.

pca_cov reconstructs a covariance matrix from its top-k principal components
plus a scalar noise floor and is tested for shape, symmetry, positive
definiteness, clipping of k, and NaN propagation.
"""

import math

import numpy as np
import polars as pl
import pytest

from basanos.math._signal import pca_cov, shrink2id, vol_adj

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


# ─── pca_cov ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_corr_matrix() -> np.ndarray:
    """A well-conditioned 4×4 correlation matrix for pca_cov tests."""
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(50, 4))
    corr_mat = np.corrcoef(raw.T)
    return corr_mat


def test_pca_cov_output_shape(sample_corr_matrix: np.ndarray) -> None:
    """pca_cov must return a matrix with the same shape as the input."""
    result = pca_cov(sample_corr_matrix, k=2)
    assert result.shape == sample_corr_matrix.shape


def test_pca_cov_is_symmetric(sample_corr_matrix: np.ndarray) -> None:
    """pca_cov result must be symmetric (up to floating-point tolerance)."""
    result = pca_cov(sample_corr_matrix, k=2)
    np.testing.assert_allclose(result, result.T, atol=1e-12)


def test_pca_cov_is_positive_definite(sample_corr_matrix: np.ndarray) -> None:
    """pca_cov result must have all eigenvalues strictly positive."""
    for k in [1, 2, 3, 4]:
        result = pca_cov(sample_corr_matrix, k=k)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals > 0), f"Not positive definite for k={k}: min eigval={eigvals.min()}"


def test_pca_cov_k_greater_than_n_clips_to_n(sample_corr_matrix: np.ndarray) -> None:
    """pca_cov must silently clip k to n when k > n."""
    n = sample_corr_matrix.shape[0]
    result_clipped = pca_cov(sample_corr_matrix, k=n + 100)
    result_exact = pca_cov(sample_corr_matrix, k=n)
    np.testing.assert_allclose(result_clipped, result_exact, atol=1e-12)


def test_pca_cov_k_equals_n_uses_all_components(sample_corr_matrix: np.ndarray) -> None:
    """pca_cov with k=n should reconstruct a full-rank positive definite matrix."""
    n = sample_corr_matrix.shape[0]
    result = pca_cov(sample_corr_matrix, k=n)
    assert result.shape == (n, n)
    eigvals = np.linalg.eigvalsh(result)
    assert np.all(eigvals > 0)


def test_pca_cov_k1_single_factor(sample_corr_matrix: np.ndarray) -> None:
    """pca_cov with k=1 produces a rank-1 factor component plus noise floor."""
    result = pca_cov(sample_corr_matrix, k=1)
    # The result is (n,n) and positive definite
    eigvals = np.linalg.eigvalsh(result)
    assert np.all(eigvals > 0)


def test_pca_cov_nan_matrix_propagates_nan() -> None:
    """pca_cov with an all-NaN input must return an all-NaN output (warmup safety)."""
    nan_mat = np.full((3, 3), np.nan)
    result = pca_cov(nan_mat, k=2)
    assert np.all(np.isnan(result))


def test_pca_cov_identity_input() -> None:
    """pca_cov on the identity matrix must return a positive-definite matrix."""
    eye_mat = np.eye(3)
    result = pca_cov(eye_mat, k=2)
    assert result.shape == (3, 3)
    eigvals = np.linalg.eigvalsh(result)
    assert np.all(eigvals > 0)
