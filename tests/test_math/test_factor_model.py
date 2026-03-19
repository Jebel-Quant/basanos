"""Tests for basanos.math._factor_model.FactorModel (Section 4.1).

Tests cover:
- Direct construction and the frozen (immutable) contract
- __post_init__ shape and positivity validation
- Properties: n_assets, n_factors, covariance
- from_returns classmethod: shapes, unit diagonal, SVD identity, edge cases
"""

import dataclasses

import numpy as np
import pytest

from basanos.exceptions import (
    FactorCountError,
    FactorCovarianceShapeError,
    FactorLoadingsDimensionError,
    IdiosyncraticVarShapeError,
    NonPositiveIdiosyncraticVarError,
    ReturnMatrixDimensionError,
)
from basanos.math._factor_model import FactorModel

# ─── helpers ─────────────────────────────────────────────────────────────────


def _make_fm(n: int = 4, k: int = 2) -> FactorModel:
    """Construct a minimal valid FactorModel with *n* assets and *k* factors."""
    rng = np.random.default_rng(42)
    b_mat = rng.standard_normal((n, k))
    f_mat = np.eye(k)
    d = np.ones(n)
    return FactorModel(factor_loadings=b_mat, factor_covariance=f_mat, idiosyncratic_var=d)


# ─── construction & frozen semantics ─────────────────────────────────────────


def test_construction_stores_fields():
    """Stored fields must equal the arrays passed at construction."""
    b_mat = np.eye(3, 2)
    f_mat = np.eye(2)
    d = np.ones(3)
    fm = FactorModel(factor_loadings=b_mat, factor_covariance=f_mat, idiosyncratic_var=d)
    np.testing.assert_array_equal(fm.factor_loadings, b_mat)
    np.testing.assert_array_equal(fm.factor_covariance, f_mat)
    np.testing.assert_array_equal(fm.idiosyncratic_var, d)


def test_frozen_raises_on_attribute_assignment():
    """Assigning to any field of a frozen dataclass must raise FrozenInstanceError."""
    fm = _make_fm()
    with pytest.raises(dataclasses.FrozenInstanceError):
        fm.factor_loadings = np.eye(4, 2)  # type: ignore[misc]


# ─── __post_init__ validation ─────────────────────────────────────────────────


def test_post_init_rejects_1d_factor_loadings():
    """factor_loadings must be 2-D; a 1-D array must raise FactorLoadingsDimensionError."""
    with pytest.raises(FactorLoadingsDimensionError, match="factor_loadings must be 2-D"):
        FactorModel(
            factor_loadings=np.ones(4),
            factor_covariance=np.eye(1),
            idiosyncratic_var=np.ones(4),
        )


def test_post_init_rejects_wrong_factor_covariance_shape():
    """factor_covariance shape must match k from factor_loadings."""
    with pytest.raises(FactorCovarianceShapeError, match="factor_covariance must have shape"):
        FactorModel(
            factor_loadings=np.eye(4, 2),
            factor_covariance=np.eye(3),  # wrong: should be (2, 2)
            idiosyncratic_var=np.ones(4),
        )


def test_post_init_rejects_wrong_idiosyncratic_var_shape():
    """idiosyncratic_var length must match n from factor_loadings."""
    with pytest.raises(IdiosyncraticVarShapeError, match="idiosyncratic_var must have shape"):
        FactorModel(
            factor_loadings=np.eye(4, 2),
            factor_covariance=np.eye(2),
            idiosyncratic_var=np.ones(5),  # wrong: should be (4,)
        )


def test_post_init_rejects_non_positive_idiosyncratic_var():
    """All entries of idiosyncratic_var must be strictly positive."""
    with pytest.raises(NonPositiveIdiosyncraticVarError, match="strictly positive"):
        FactorModel(
            factor_loadings=np.eye(3, 2),
            factor_covariance=np.eye(2),
            idiosyncratic_var=np.array([1.0, 0.0, 1.0]),  # zero entry
        )


def test_post_init_rejects_negative_idiosyncratic_var():
    """Negative entries in idiosyncratic_var must also raise NonPositiveIdiosyncraticVarError."""
    with pytest.raises(NonPositiveIdiosyncraticVarError, match="strictly positive"):
        FactorModel(
            factor_loadings=np.eye(3, 2),
            factor_covariance=np.eye(2),
            idiosyncratic_var=np.array([1.0, -0.5, 1.0]),
        )


# ─── n_assets and n_factors properties ───────────────────────────────────────


def test_n_assets_matches_loadings_rows():
    """n_assets must equal the number of rows in factor_loadings."""
    for n in (2, 5, 10):
        fm = _make_fm(n=n, k=2)
        assert fm.n_assets == n


def test_n_factors_matches_loadings_columns():
    """n_factors must equal the number of columns in factor_loadings."""
    for k in (1, 2, 4):
        fm = _make_fm(n=6, k=k)
        assert fm.n_factors == k


# ─── covariance property ─────────────────────────────────────────────────────


def test_covariance_shape():
    """Covariance must return an (n, n) array."""
    fm = _make_fm(n=5, k=2)
    assert fm.covariance.shape == (5, 5)


def test_covariance_matches_formula():
    """Covariance must equal B*F*B^T + diag(d) exactly."""
    b_mat = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    f_mat = np.array([[2.0, 0.0], [0.0, 3.0]])
    d = np.array([0.5, 0.5, 1.0])
    fm = FactorModel(factor_loadings=b_mat, factor_covariance=f_mat, idiosyncratic_var=d)
    expected = b_mat @ f_mat @ b_mat.T + np.diag(d)
    np.testing.assert_allclose(fm.covariance, expected)


def test_covariance_is_symmetric():
    """The reconstructed covariance matrix must be symmetric."""
    fm = _make_fm(n=6, k=3)
    cov = fm.covariance
    np.testing.assert_allclose(cov, cov.T, atol=1e-14)


def test_covariance_diagonal_equals_factor_plus_idio():
    """Each diagonal entry of covariance must equal (B*F*B^T)_ii + d_i."""
    fm = _make_fm(n=5, k=2)
    systematic_diag = np.einsum("ij,jj,ij->i", fm.factor_loadings, fm.factor_covariance, fm.factor_loadings)
    expected_diag = systematic_diag + fm.idiosyncratic_var
    np.testing.assert_allclose(fm.covariance.diagonal(), expected_diag, atol=1e-14)


# ─── from_returns classmethod ─────────────────────────────────────────────────


@pytest.fixture
def return_matrix() -> np.ndarray:
    """50 x 5 standard-normal return matrix."""
    return np.random.default_rng(7).standard_normal((50, 5))


def test_from_returns_shapes(return_matrix):
    """from_returns must produce a FactorModel with n=5, k=2."""
    fm = FactorModel.from_returns(return_matrix, k=2)
    assert fm.n_assets == 5
    assert fm.n_factors == 2
    assert fm.factor_loadings.shape == (5, 2)
    assert fm.factor_covariance.shape == (2, 2)
    assert fm.idiosyncratic_var.shape == (5,)


def test_from_returns_covariance_unit_diagonal(return_matrix):
    """Covariance from from_returns must have strictly positive diagonal entries."""
    fm = FactorModel.from_returns(return_matrix, k=2)
    diag = fm.covariance.diagonal()
    assert np.all(diag > 0)


def test_from_returns_idiosyncratic_var_all_positive(return_matrix):
    """All idiosyncratic_var entries must be strictly positive."""
    fm = FactorModel.from_returns(return_matrix, k=3)
    assert np.all(fm.idiosyncratic_var > 0)


def test_from_returns_factor_covariance_is_diagonal(return_matrix):
    """Factor covariance from SVD must be diagonal (off-diagonals are zero)."""
    fm = FactorModel.from_returns(return_matrix, k=2)
    off_diag = fm.factor_covariance - np.diag(np.diag(fm.factor_covariance))
    np.testing.assert_allclose(off_diag, np.zeros_like(off_diag), atol=1e-14)


def test_from_returns_factor_covariance_entries_positive(return_matrix):
    """Diagonal entries of factor_covariance must all be strictly positive."""
    fm = FactorModel.from_returns(return_matrix, k=2)
    assert np.all(np.diag(fm.factor_covariance) > 0)


def test_from_returns_k_equals_one(return_matrix):
    """from_returns must work for k=1 (single-factor / market model)."""
    fm = FactorModel.from_returns(return_matrix, k=1)
    assert fm.n_factors == 1
    assert fm.factor_loadings.shape == (5, 1)


def test_from_returns_k_equals_min_t_n():
    """from_returns must work when k equals min(T, n) (full rank)."""
    ret_mat = np.random.default_rng(0).standard_normal((10, 4))
    fm = FactorModel.from_returns(ret_mat, k=4)
    assert fm.n_factors == 4


def test_from_returns_raises_on_1d_input():
    """from_returns must raise ReturnMatrixDimensionError when input is not 2-D."""
    with pytest.raises(ReturnMatrixDimensionError, match="Return matrix must be 2-D"):
        FactorModel.from_returns(np.ones(10), k=1)


def test_from_returns_raises_on_k_zero():
    """from_returns must raise FactorCountError when k < 1."""
    ret_mat = np.random.default_rng(0).standard_normal((20, 4))
    with pytest.raises(FactorCountError, match="k must satisfy"):
        FactorModel.from_returns(ret_mat, k=0)


def test_from_returns_raises_on_k_too_large():
    """from_returns must raise FactorCountError when k > min(T, n)."""
    ret_mat = np.random.default_rng(0).standard_normal((20, 4))
    with pytest.raises(FactorCountError, match="k must satisfy"):
        FactorModel.from_returns(ret_mat, k=5)


def test_from_returns_public_api_import():
    """FactorModel must be importable from the basanos.math public namespace."""
    from basanos.math import FactorModel as PublicFactorModel

    assert PublicFactorModel is FactorModel
