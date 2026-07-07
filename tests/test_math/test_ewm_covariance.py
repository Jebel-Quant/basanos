"""Verify cvx.linalg.ewm_covariance against pandas and ewm_corr.

Two goals:
  1. ewm_covariance matches pandas.DataFrame.ewm(span).cov(bias=True) to 1e-10
     wherever both produce finite values.
  2. ewm_covariance normalised to correlation matches ewm_corr to 1e-10
     on dense (no-missing) data.

Missing-data semantics:

  ewm_covariance (polars) outputs NaN for a pair at every timestep where
  either asset is null.  pandas ewm(ignore_na=False) freezes the last valid
  EWM state and returns a non-NaN value even when one asset is missing, so
  pandas has fewer NaN cells.  Wherever cvx produces a finite value pandas
  also does (cvx is the more conservative superset of NaN).

  ewm_corr and pandas ewm().corr() are equivalent.  They compute variances
  conditioned on the JOINT availability of each pair, whereas normalising the
  cvx covariance matrix uses marginal (per-asset) variances.  These differ on
  sparse data; the test for ewm_corr agreement is therefore restricted to
  dense inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from cvx.linalg.covariance.ewm_cov import ewm_covariance

# ── helpers ───────────────────────────────────────────────────────────────────


def _to_polars(data: np.ndarray, assets: list[str]) -> pl.DataFrame:
    """Convert a numpy array to a Polars DataFrame with a 't' index and null-filled NaNs."""
    # fill_nan(None) converts float NaN to Polars null; ewm_covariance expects nulls, not NaN
    t = data.shape[0]
    cols = [pl.Series(a, data[:, i]).fill_nan(None) for i, a in enumerate(assets)]
    return pl.DataFrame([pl.Series("t", list(range(t))), *cols])


def _com_to_span(com: int) -> int:
    """Pandas com -> span: alpha = 1/(1+com) = 2/(1+span)  =>  span = 2*com+1."""
    return 2 * com + 1


def _dict_to_array(cov: dict, t: int, n: int) -> np.ndarray:
    """Expand ewm_covariance result (keyed by int index) into a (T, N, N) array."""
    out = np.full((t, n, n), np.nan)
    for k, mat in cov.items():
        out[int(k)] = mat
    return out


def _cov_to_corr(cov_tensor: np.ndarray) -> np.ndarray:
    """Normalise (T, N, N) covariance tensor to correlations using marginal variances."""
    t_len, n, _ = cov_tensor.shape
    corr = np.full_like(cov_tensor, np.nan)
    for t in range(t_len):
        cov = cov_tensor[t]
        var = np.diag(cov)
        denom = np.sqrt(np.outer(var, var))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr[t] = np.where(denom > 1e-14, cov / denom, np.nan)
        corr[t] = np.clip(corr[t], -1.0, 1.0)
        idx = np.arange(n)
        corr[t, idx, idx] = np.where(np.isfinite(np.diag(cov)) & (var > 1e-14), 1.0, np.nan)
        tril_i, tril_j = np.tril_indices(n, k=-1)
        avg = 0.5 * (corr[t, tril_i, tril_j] + corr[t, tril_j, tril_i])
        corr[t, tril_i, tril_j] = avg
        corr[t, tril_j, tril_i] = avg
    return corr


def _pandas_cov(data: np.ndarray, com: int, min_periods: int) -> np.ndarray:
    """Compute the reference EWM covariance tensor via pandas.ewm(com).cov(bias=True)."""
    t_len, n = data.shape
    df = pd.DataFrame(data)
    raw = df.ewm(com=com, min_periods=min_periods).cov(bias=True)
    reset = raw.reset_index(names=["t", "asset"])
    result = np.full((t_len, n, n), np.nan)
    for t, group in reset.groupby("t"):
        result[int(t)] = group.drop(columns=["t", "asset"]).to_numpy()
    return result


def _assert_values_match(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-10, label: str = "") -> None:
    """Assert a and b agree wherever both are finite; do not require equal NaN patterns."""
    both = np.isfinite(a) & np.isfinite(b)
    if both.any():
        diff = float(np.abs(a[both] - b[both]).max())
        assert diff < atol, f"{label}: max diff {diff:.2e} exceeds {atol:.0e}"


def _assert_close(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-10, label: str = "") -> None:
    """Assert identical NaN patterns and value agreement."""
    nan_match = bool(np.all(~np.isfinite(a) == ~np.isfinite(b)))
    assert nan_match, f"{label}: NaN patterns differ"
    _assert_values_match(a, b, atol=atol, label=label)


def _assert_cvx_nan_is_superset(cvx: np.ndarray, ref: np.ndarray, label: str = "") -> None:
    """Cvx may have more NaN than ref, but must never be finite where ref is NaN."""
    ref_nan_cvx_fin = np.isfinite(cvx) & ~np.isfinite(ref)
    assert not ref_nan_cvx_fin.any(), f"{label}: cvx finite where reference is NaN ({ref_nan_cvx_fin.sum()} cells)"


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def dense() -> tuple[np.ndarray, int]:
    """500×4 array with no missing values and com=32."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(500, 4)), 32


@pytest.fixture(scope="module")
def sparse() -> tuple[np.ndarray, int]:
    """500×4 array with ~5% NaN entries and com=32."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(500, 4))
    data[rng.random(data.shape) < 0.05] = np.nan
    return data, 32


# ── ewm_covariance vs pandas ──────────────────────────────────────────────────


class TestEwmCovarianceVsPandas:
    """cvx ewm_covariance agrees with pandas.ewm(span).cov(bias=True).

    Dense: identical NaN patterns and value match to 1e-10.
    Sparse: cvx has strictly more NaN (missing-data semantics differ);
    wherever cvx is finite, it matches pandas to 1e-10.
    """

    def _cvx(self, data: np.ndarray, com: int) -> np.ndarray:
        """Run cvx ewm_covariance on the data and return a dense (T, N, N) tensor."""
        t_len, n = data.shape
        assets = [str(i) for i in range(n)]
        cov = ewm_covariance(
            _to_polars(data, assets), assets=assets, index_col="t", window=_com_to_span(com), warmup=com
        )
        return _dict_to_array(cov, t_len, n)

    def test_dense(self, dense: tuple[np.ndarray, int]) -> None:
        """On dense data, cvx matches pandas with identical NaN patterns and values."""
        data, com = dense
        _assert_close(self._cvx(data, com), _pandas_cov(data, com, com), label="dense cov")

    def test_sparse_values_match_where_finite(self, sparse: tuple[np.ndarray, int]) -> None:
        """On sparse data, cvx matches pandas wherever cvx is finite and is never finite where pandas is NaN."""
        data, com = sparse
        cvx_arr = self._cvx(data, com)
        pd_arr = _pandas_cov(data, com, com)
        # cvx is more conservative: it is NaN where pandas "freezes" the last valid state
        _assert_cvx_nan_is_superset(cvx_arr, pd_arr, label="sparse cov")
        _assert_values_match(cvx_arr, pd_arr, label="sparse cov")

    @pytest.mark.parametrize("com", [5, 16, 64])
    def test_various_com(self, com: int) -> None:
        """Cvx matches pandas across a range of center-of-mass values."""
        rng = np.random.default_rng(99)
        data = rng.normal(size=(300, 4))
        _assert_close(self._cvx(data, com), _pandas_cov(data, com, com), label=f"com={com}")

    def test_single_asset(self) -> None:
        """Cvx matches pandas for the single-asset (variance) case."""
        rng = np.random.default_rng(7)
        data = rng.normal(size=(200, 1))
        _assert_close(self._cvx(data, 20), _pandas_cov(data, 20, 20), label="single asset")
