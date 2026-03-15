"""Tests for correlation matrix construction with staggered asset availability.

Scenario
--------
4 years (2020-2023), 4 assets with staggered entry/exit:

    asset_1 : available in years 1-2 only  (null from 2022-01-01 onward)
    asset_2 : available for all 4 years
    asset_3 : enters in year 2             (null before 2021-01-01)
    asset_4 : enters in year 3             (null before 2022-01-01)

These tests validate that ``BasanosEngine.cor`` correctly reflects asset
availability at every timestamp and that ``BasanosEngine.cash_position``
produces NaN positions precisely where prices are absent.
"""

from __future__ import annotations

import pathlib
import tempfile
from datetime import date

import numpy as np
import polars as pl
import pytest

from basanos.math import BasanosConfig, BasanosEngine

# ---------------------------------------------------------------------------
# Year boundary constants
# ---------------------------------------------------------------------------
_Y1 = date(2020, 1, 1)  # start of year 1
_Y2 = date(2021, 1, 1)  # start of year 2
_Y3 = date(2022, 1, 1)  # start of year 3
_Y4 = date(2023, 1, 1)  # start of year 4
_END = date(2023, 12, 31)  # end of year 4

# Asset column index positions (match insertion order in the fixture)
_IDX_1, _IDX_2, _IDX_3, _IDX_4 = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mask(series: np.ndarray, dates: list[date], *, from_: date | None = None, before: date | None = None) -> list:
    """Return a list of floats with ``None`` outside the [from_, before) window."""
    return [
        float(v) if (from_ is None or d >= from_) and (before is None or d < before) else None
        for d, v in zip(dates, series, strict=False)
    ]


@pytest.fixture(scope="module")
def prices() -> pl.DataFrame:
    """Four-year, four-asset price frame with staggered availability.

    Availability:
        asset_1 : years 1-2 (null from year 3 onward)
        asset_2 : all 4 years
        asset_3 : years 2-4 (null in year 1)
        asset_4 : years 3-4 (null in years 1-2)
    """
    dates_series = pl.date_range(start=_Y1, end=_END, interval="1d", eager=True)
    n = len(dates_series)
    rng = np.random.default_rng(42)
    base = {k: 100.0 + np.cumsum(rng.normal(0.0, 0.3, size=n)) for k in range(1, 5)}
    dl = dates_series.to_list()

    return pl.DataFrame(
        {
            "date": dates_series,
            "asset_1": pl.Series(_mask(base[1], dl, before=_Y3), dtype=pl.Float64),
            "asset_2": pl.Series(list(base[2].astype(float)), dtype=pl.Float64),
            "asset_3": pl.Series(_mask(base[3], dl, from_=_Y2), dtype=pl.Float64),
            "asset_4": pl.Series(_mask(base[4], dl, from_=_Y3), dtype=pl.Float64),
        }
    )


@pytest.fixture(scope="module")
def mu(prices: pl.DataFrame) -> pl.DataFrame:
    """Simple bounded expected-return signal aligned with *prices*."""
    n = prices.height
    theta = np.linspace(0.0, 4.0 * np.pi, num=n)
    return pl.DataFrame(
        {
            "date": prices["date"],
            "asset_1": pl.Series(np.tanh(np.sin(theta)), dtype=pl.Float64),
            "asset_2": pl.Series(np.tanh(np.cos(theta)), dtype=pl.Float64),
            "asset_3": pl.Series(np.tanh(np.sin(2.0 * theta)), dtype=pl.Float64),
            "asset_4": pl.Series(np.tanh(np.cos(2.0 * theta)), dtype=pl.Float64),
        }
    )


@pytest.fixture(scope="module")
def cfg() -> BasanosConfig:
    """Basanos config with moderate EWMA windows suitable for 4-year data."""
    return BasanosConfig(vola=16, corr=32, clip=3.5, shrink=0.5, aum=1e6)


@pytest.fixture(scope="module")
def cor(prices: pl.DataFrame, mu: pl.DataFrame, cfg: BasanosConfig) -> dict:
    """Pre-computed correlation matrices for the full 4-year scenario."""
    return BasanosEngine(prices=prices, mu=mu, cfg=cfg).cor


# ---------------------------------------------------------------------------
# Structural / shape tests
# ---------------------------------------------------------------------------


class TestCorMatrixShape:
    """Structural properties of the returned correlation dict."""

    def test_returns_dict(self, cor: dict) -> None:
        """Cor should return a plain Python dict."""
        assert isinstance(cor, dict)

    def test_has_one_entry_per_day(self, prices: pl.DataFrame, cor: dict) -> None:
        """The dict should have exactly one entry for each row in the prices frame."""
        assert len(cor) == prices.height

    def test_each_matrix_is_4x4_numpy_array(self, cor: dict) -> None:
        """Every value in the dict should be a (4, 4) NumPy ndarray."""
        for _d, mat in cor.items():
            assert isinstance(mat, np.ndarray)
            assert mat.shape == (4, 4), f"Expected (4, 4), got {mat.shape} at {_d}"

    def test_finite_correlations_are_in_unit_interval(self, cor: dict) -> None:
        """Finite correlation values must lie within [−1, 1]."""
        for _d, mat in cor.items():
            finite = mat[np.isfinite(mat)]
            if finite.size:
                assert np.all(finite >= -1.0 - 1e-9), f"Correlation below -1 at {_d}"
                assert np.all(finite <= 1.0 + 1e-9), f"Correlation above +1 at {_d}"

    def test_matrices_are_symmetric(self, cor: dict) -> None:
        """Sample every 30th date to keep the test fast."""
        for d in list(cor.keys())[::30]:
            mat = cor[d]
            both_finite = np.isfinite(mat) & np.isfinite(mat.T)
            if both_finite.any():
                np.testing.assert_allclose(
                    mat[both_finite],
                    mat.T[both_finite],
                    atol=1e-10,
                    err_msg=f"Matrix not symmetric at {d}",
                )

    def test_diagonal_is_one_where_finite(self, cor: dict, cfg: BasanosConfig) -> None:
        """After a generous warmup period, every finite diagonal entry equals 1."""
        warmup = cfg.corr + cfg.vola
        for d in list(cor.keys())[warmup::30]:
            mat = cor[d]
            for i, val in enumerate(np.diag(mat)):
                if np.isfinite(val):
                    assert abs(val - 1.0) < 1e-6, f"Diagonal [{i},{i}] = {val} at {d}; expected 1.0"


# ---------------------------------------------------------------------------
# Asset-availability tests
# ---------------------------------------------------------------------------


class TestCorMatrixAssetAvailability:
    """The correlation matrices should reflect which assets have price data."""

    def test_asset3_and_asset4_rows_are_nan_in_year1(self, cor: dict, cfg: BasanosConfig) -> None:
        """asset_3 and asset_4 have no price data in year 1.

        After the initial EWM warmup for assets 1 and 2, the rows *and* columns
        corresponding to asset_3 (index 2) and asset_4 (index 3) must be entirely
        non-finite because no observations exist yet.
        """
        year1_post_warmup = [d for d in cor if d.year == 2020][cfg.corr :]
        if not year1_post_warmup:
            pytest.skip("Not enough year-1 dates after warmup")

        for d in year1_post_warmup[:10]:
            mat = cor[d]
            assert not np.any(np.isfinite(mat[_IDX_3, :])), f"asset_3 row not all-NaN at {d}"
            assert not np.any(np.isfinite(mat[:, _IDX_3])), f"asset_3 col not all-NaN at {d}"
            assert not np.any(np.isfinite(mat[_IDX_4, :])), f"asset_4 row not all-NaN at {d}"
            assert not np.any(np.isfinite(mat[:, _IDX_4])), f"asset_4 col not all-NaN at {d}"

    def test_asset1_asset2_cross_correlation_finite_in_late_year1(self, cor: dict, cfg: BasanosConfig) -> None:
        """Both asset_1 and asset_2 are present in year 1.

        After the EWM warmup their cross-correlation should be finite.
        """
        warmup = cfg.corr + cfg.vola + 10
        late_year1 = sorted(d for d in cor if d.year == 2020)[warmup:]
        assert late_year1, "Need year-1 dates beyond warmup"

        for d in late_year1[-5:]:
            mat = cor[d]
            assert np.isfinite(mat[_IDX_1, _IDX_2]), f"asset_1 vs asset_2 not finite at {d}"
            assert np.isfinite(mat[_IDX_2, _IDX_1]), f"asset_2 vs asset_1 not finite at {d}"

    def test_asset3_correlations_become_finite_in_late_year2(self, cor: dict, cfg: BasanosConfig) -> None:
        """asset_3 enters year 2.

        After its own EWM warmup, correlations with asset_2 (always present) should be finite.
        """
        warmup_after_entry = cfg.corr + 10
        late_year2 = sorted(d for d in cor if d.year == 2021)[warmup_after_entry:]
        assert late_year2, "Need year-2 dates beyond warmup"

        for d in late_year2[-5:]:
            mat = cor[d]
            assert np.isfinite(mat[_IDX_2, _IDX_3]), f"asset_2 vs asset_3 not finite at {d}"
            assert np.isfinite(mat[_IDX_3, _IDX_2]), f"asset_3 vs asset_2 not finite at {d}"

    def test_asset4_correlations_become_finite_in_late_year3(self, cor: dict, cfg: BasanosConfig) -> None:
        """asset_4 enters year 3.

        After its EWM warmup, correlations with asset_2 and asset_3 should be finite.
        """
        warmup_after_entry = cfg.corr + 10
        late_year3 = sorted(d for d in cor if d.year == 2022)[warmup_after_entry:]
        assert late_year3, "Need year-3 dates beyond warmup"

        for d in late_year3[-5:]:
            mat = cor[d]
            assert np.isfinite(mat[_IDX_2, _IDX_4]), f"asset_2 vs asset_4 not finite at {d}"
            assert np.isfinite(mat[_IDX_4, _IDX_2]), f"asset_4 vs asset_2 not finite at {d}"
            assert np.isfinite(mat[_IDX_3, _IDX_4]), f"asset_3 vs asset_4 not finite at {d}"
            assert np.isfinite(mat[_IDX_4, _IDX_3]), f"asset_4 vs asset_3 not finite at {d}"

    def test_active_assets_have_finite_diagonal_in_late_year3(self, cor: dict, cfg: BasanosConfig) -> None:
        """In year 3, asset_2, asset_3, and asset_4 all have live price data.

        Note: asset_1 price is null from year 3 onward (only EWM memory may
        persist, so its diagonal is not checked here).  For the three assets
        with active prices, every diagonal entry should equal 1.0 after the
        respective warmup periods.
        """
        warmup_after_entry = cfg.corr + cfg.vola + 10
        late_year3 = sorted(d for d in cor if d.year == 2022)[warmup_after_entry:]
        if not late_year3:
            pytest.skip("Not enough year-3 dates after warmup")

        for d in late_year3[-5:]:
            mat = cor[d]
            for idx in (_IDX_2, _IDX_3, _IDX_4):
                diag = mat[idx, idx]
                assert np.isfinite(diag), f"Diagonal [{idx},{idx}] is NaN at {d}"
                assert abs(diag - 1.0) < 1e-6, f"Diagonal [{idx},{idx}] = {diag} at {d}"


# ---------------------------------------------------------------------------
# Cash-position integration tests
# ---------------------------------------------------------------------------


class TestCashPositionWithStaggeredAssets:
    """BasanosEngine.cash_position must be NaN exactly where prices are absent."""

    @pytest.fixture(scope="class")
    def cash_pos(self, prices: pl.DataFrame, mu: pl.DataFrame, cfg: BasanosConfig) -> pl.DataFrame:
        """Build the cash_position DataFrame once per test class."""
        return BasanosEngine(prices=prices, mu=mu, cfg=cfg).cash_position

    def test_schema_matches_prices(self, cash_pos: pl.DataFrame, prices: pl.DataFrame) -> None:
        """cash_position should have the same row count and columns as prices."""
        assert isinstance(cash_pos, pl.DataFrame)
        assert cash_pos.height == prices.height
        assert cash_pos.columns[0] == "date"
        assert set(cash_pos.columns[1:]) == {"asset_1", "asset_2", "asset_3", "asset_4"}

    def test_asset1_positions_nan_from_year3(self, cash_pos: pl.DataFrame) -> None:
        """asset_1 price is null from year 3 onward; positions must be NaN."""
        dates = cash_pos["date"].to_list()
        vals = cash_pos["asset_1"].to_numpy()
        year3_plus = [i for i, d in enumerate(dates) if d >= _Y3]
        for i in year3_plus[:20]:
            assert not np.isfinite(vals[i]), f"Expected NaN for asset_1 at {dates[i]}, got {vals[i]}"

    def test_asset3_positions_nan_in_year1(self, cash_pos: pl.DataFrame) -> None:
        """asset_3 price is null in year 1; positions must be NaN."""
        dates = cash_pos["date"].to_list()
        vals = cash_pos["asset_3"].to_numpy()
        year1 = [i for i, d in enumerate(dates) if d < _Y2]
        for i in year1:
            assert not np.isfinite(vals[i]), f"Expected NaN for asset_3 at {dates[i]}, got {vals[i]}"

    def test_asset4_positions_nan_in_years_1_and_2(self, cash_pos: pl.DataFrame) -> None:
        """asset_4 price is null in years 1-2; positions must be NaN."""
        dates = cash_pos["date"].to_list()
        vals = cash_pos["asset_4"].to_numpy()
        years12 = [i for i, d in enumerate(dates) if d < _Y3]
        for i in years12:
            assert not np.isfinite(vals[i]), f"Expected NaN for asset_4 at {dates[i]}, got {vals[i]}"

    def test_asset2_positions_finite_after_warmup(self, cash_pos: pl.DataFrame, cfg: BasanosConfig) -> None:
        """asset_2 is present for all 4 years; after warmup its positions are finite."""
        warmup = cfg.corr + cfg.vola
        tail_vals = cash_pos["asset_2"].to_numpy()[warmup:]
        assert np.all(np.isfinite(tail_vals)), "asset_2 has non-finite positions after warmup"


# ---------------------------------------------------------------------------
# Tensor / flat-file round-trip tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tensor_engine(prices: pl.DataFrame, mu: pl.DataFrame, cfg: BasanosConfig) -> BasanosEngine:
    """A single BasanosEngine instance shared across the tensor test class."""
    return BasanosEngine(prices=prices, mu=mu, cfg=cfg)


class TestCorTensorFlatFile:
    """Store all correlation matrices in a tensor and round-trip via a flat file."""

    def test_tensor_shape_and_type(self, tensor_engine: BasanosEngine, prices: pl.DataFrame) -> None:
        """cor_tensor should return a 3-D NumPy ndarray of shape (T, N, N)."""
        tensor = tensor_engine.cor_tensor
        n_assets = len(tensor_engine.assets)
        assert isinstance(tensor, np.ndarray)
        assert tensor.ndim == 3
        assert tensor.shape == (prices.height, n_assets, n_assets)

    def test_tensor_slices_match_cor_dict(self, tensor_engine: BasanosEngine) -> None:
        """Each slice tensor[t] must equal the corresponding matrix in cor."""
        tensor = tensor_engine.cor_tensor
        for t, mat in enumerate(tensor_engine.cor.values()):
            np.testing.assert_array_equal(
                tensor[t],
                mat,
                err_msg=f"tensor[{t}] does not match cor dict entry at index {t}",
            )

    def test_tensor_saves_and_loads_from_flat_file(self, tensor_engine: BasanosEngine) -> None:
        """Save the tensor to a .npy flat file and reload it; values must be identical."""
        tensor = tensor_engine.cor_tensor
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "cor_tensor.npy"
            np.save(path, tensor)
            loaded = np.load(path)

        assert loaded.shape == tensor.shape
        np.testing.assert_array_equal(loaded, tensor)

    def test_tensor_reproduced_from_flat_file_matches_cor_dict(self, tensor_engine: BasanosEngine) -> None:
        """After a save/load round-trip the tensor must reproduce every cor dict entry."""
        tensor = tensor_engine.cor_tensor
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "cor_tensor.npy"
            np.save(path, tensor)
            loaded = np.load(path)

        for t, mat in enumerate(tensor_engine.cor.values()):
            np.testing.assert_array_equal(
                loaded[t],
                mat,
                err_msg=f"Reproduced tensor[{t}] does not match original cor dict entry",
            )
