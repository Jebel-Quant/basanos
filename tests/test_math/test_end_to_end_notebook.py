"""CI execution gate and API contract tests for the end_to_end notebook.

This module drives ``book/marimo/notebooks/end_to_end.py`` directly via
``app.run()`` so that any change to the notebook's data-generation cells,
configuration, or engine construction is automatically reflected here.
No constants or logic are duplicated from the notebook.

Covered outputs:

- ``prices``         — synthetic multi-sector price DataFrame shape and schema
- ``mu``             — momentum signal DataFrame shape
- ``cfg``            — ``BasanosConfig`` parameter values (slider defaults)
- ``engine``         — ``BasanosEngine`` cash-position shape and asset list
- ``position_status``— per-row status label breakdown (valid / degenerate / zero_signal)
- IC / ICIR          — information coefficient and information ratio are finite
- ``portfolio``      — final NAV is positive and deterministic (fixed seed 2024)
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from basanos.math import BasanosConfig, BasanosEngine

_NOTEBOOK = Path(__file__).parents[2] / "book/marimo/notebooks/end_to_end.py"

_EXPECTED_ASSETS = [
    "Technology",
    "Healthcare",
    "Financials",
    "Energy",
    "Consumer",
    "Industrials",
    "Materials",
    "Utilities",
]
_N_ASSETS = len(_EXPECTED_ASSETS)
_N_DAYS = 1_260
# Tolerance for day-1 NAV vs AUM: P&L on day 1 is the return on the first day's
# position.  With 10M AUM and moderate daily returns (≤ few percent), NAV stays
# within 5% of AUM on the opening day of trading.
_INITIAL_NAV_TOLERANCE = 0.05


# ─── Notebook execution fixture ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def notebook_defs() -> Mapping[str, Any]:
    """Run the end_to_end notebook once via app.run() and return all cell definitions.

    This is the single source of truth for all fixtures in this module.
    Any change to the notebook (data generation, config, engine construction)
    is automatically picked up here — no mirrored constants to drift.
    """
    sys.path.insert(0, str(_NOTEBOOK.parent))
    try:
        from end_to_end import app  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    _outputs, defs = app.run()
    return defs


@pytest.fixture(scope="module")
def notebook_prices(notebook_defs: Mapping[str, Any]) -> pl.DataFrame:
    """Prices DataFrame as produced by cell_04 of the end_to_end notebook."""
    return notebook_defs["prices"]


@pytest.fixture(scope="module")
def notebook_mu(notebook_defs: Mapping[str, Any]) -> pl.DataFrame:
    """Momentum signal DataFrame as produced by cell_04 of the end_to_end notebook."""
    return notebook_defs["mu"]


@pytest.fixture(scope="module")
def notebook_cfg(notebook_defs: Mapping[str, Any]) -> BasanosConfig:
    """BasanosConfig as constructed by cell_09 of the end_to_end notebook."""
    return notebook_defs["cfg"]


@pytest.fixture(scope="module")
def notebook_engine(notebook_defs: Mapping[str, Any]) -> BasanosEngine:
    """BasanosEngine as constructed by cell_12 of the end_to_end notebook."""
    return notebook_defs["engine"]


@pytest.fixture(scope="module")
def notebook_portfolio(notebook_defs: Mapping[str, Any]) -> Any:
    """Portfolio as constructed by cell_12 of the end_to_end notebook."""
    return notebook_defs["portfolio"]


# ─── Input data (prices and mu) ──────────────────────────────────────────────


class TestInputData:
    """Prices and momentum signal DataFrames match the notebook's specification."""

    def test_prices_row_count(self, notebook_prices: pl.DataFrame) -> None:
        """Prices must have exactly 1 260 rows (≈5 years of daily data)."""
        assert notebook_prices.height == _N_DAYS

    def test_prices_column_count(self, notebook_prices: pl.DataFrame) -> None:
        """Prices must have date + 8 sector columns."""
        assert notebook_prices.width == _N_ASSETS + 1

    def test_prices_column_names(self, notebook_prices: pl.DataFrame) -> None:
        """Prices must have exactly the expected sector columns."""
        assert notebook_prices.columns == ["date", *_EXPECTED_ASSETS]

    def test_prices_all_positive(self, notebook_prices: pl.DataFrame) -> None:
        """All synthetic price series must be strictly positive."""
        for asset in _EXPECTED_ASSETS:
            assert notebook_prices[asset].min() > 0, f"Non-positive price found for {asset}"

    def test_mu_row_count(self, notebook_mu: pl.DataFrame) -> None:
        """Momentum signal must have the same number of rows as prices."""
        assert notebook_mu.height == _N_DAYS

    def test_mu_column_count(self, notebook_mu: pl.DataFrame) -> None:
        """Momentum signal must have date + 8 sector columns."""
        assert notebook_mu.width == _N_ASSETS + 1

    def test_mu_bounded(self, notebook_mu: pl.DataFrame) -> None:
        """Momentum signal (tanh output) must be in [-1, 1] for all assets."""
        for asset in _EXPECTED_ASSETS:
            col = notebook_mu[asset]
            assert col.min() >= -1.0 - 1e-9, f"mu < -1 found for {asset}"
            assert col.max() <= 1.0 + 1e-9, f"mu > +1 found for {asset}"


# ─── Config ───────────────────────────────────────────────────────────────────


class TestConfig:
    """BasanosConfig matches the slider defaults wired into the notebook."""

    def test_vola(self, notebook_cfg: BasanosConfig) -> None:
        """Vola must equal the slider default of 16."""
        assert notebook_cfg.vola == 16

    def test_corr(self, notebook_cfg: BasanosConfig) -> None:
        """Corr must equal the slider default of 60."""
        assert notebook_cfg.corr == 60

    def test_clip(self, notebook_cfg: BasanosConfig) -> None:
        """Clip must equal the slider default of 4.0."""
        assert notebook_cfg.clip == 4.0

    def test_shrink(self, notebook_cfg: BasanosConfig) -> None:
        """Shrink must equal the slider default of 0.6."""
        assert notebook_cfg.shrink == pytest.approx(0.6)

    def test_aum(self, notebook_cfg: BasanosConfig) -> None:
        """AUM must equal the notebook's hard-coded value of 10 000 000."""
        assert notebook_cfg.aum == pytest.approx(10_000_000.0)


# ─── Cash position ────────────────────────────────────────────────────────────


class TestCashPosition:
    """cash_position shape and content match the notebook specification."""

    def test_row_count(self, notebook_engine: BasanosEngine, notebook_prices: pl.DataFrame) -> None:
        """cash_position must have one row per price timestamp."""
        assert notebook_engine.cash_position.height == notebook_prices.height

    def test_column_count(self, notebook_engine: BasanosEngine) -> None:
        """cash_position must have date + 8 asset columns."""
        assert notebook_engine.cash_position.width == _N_ASSETS + 1

    def test_asset_list(self, notebook_engine: BasanosEngine) -> None:
        """Engine must expose exactly the eight sector assets."""
        assert notebook_engine.assets == _EXPECTED_ASSETS

    def test_cash_position_columns(self, notebook_engine: BasanosEngine) -> None:
        """cash_position columns must match prices columns (date + assets)."""
        assert notebook_engine.cash_position.columns == ["date", *_EXPECTED_ASSETS]


# ─── Position status ─────────────────────────────────────────────────────────


class TestPositionStatus:
    """position_status output matches the expected schema and semantic rules."""

    def test_columns(self, notebook_engine: BasanosEngine) -> None:
        """position_status must have exactly ['date', 'status'] columns."""
        assert notebook_engine.position_status.columns == ["date", "status"]

    def test_row_count(self, notebook_engine: BasanosEngine, notebook_prices: pl.DataFrame) -> None:
        """position_status must have one row per price timestamp."""
        assert notebook_engine.position_status.height == notebook_prices.height

    def test_valid_status_codes(self, notebook_engine: BasanosEngine) -> None:
        """Every status value must be one of the four defined codes."""
        known = {"warmup", "zero_signal", "degenerate", "valid"}
        actual = set(notebook_engine.position_status["status"].unique().to_list())
        assert actual.issubset(known), f"Unexpected status codes: {actual - known}"

    def test_valid_row_count(self, notebook_engine: BasanosEngine) -> None:
        """With the fixed seed and default config, exactly 1 200 rows must be 'valid'."""
        valid_n = notebook_engine.position_status.filter(pl.col("status") == "valid").height
        assert valid_n == 1_200

    def test_has_valid_rows_in_tail(self, notebook_engine: BasanosEngine) -> None:
        """After EWMA convergence, the final 200 rows must include 'valid' rows."""
        tail_statuses = notebook_engine.position_status.tail(200)["status"].to_list()
        assert "valid" in tail_statuses

    def test_early_rows_not_valid(self, notebook_engine: BasanosEngine) -> None:
        """The first corr rows must not be 'valid' (EWMA not yet converged)."""
        early = notebook_engine.position_status.head(notebook_engine.cfg.corr)["status"].to_list()
        assert "valid" not in early, f"Unexpected 'valid' rows in EWMA warmup period; found statuses: {set(early)}"

    def test_no_warmup_status_in_ewma_mode(self, notebook_engine: BasanosEngine) -> None:
        """EWMA mode must not emit the 'warmup' code — that is SW-mode only."""
        statuses = set(notebook_engine.position_status["status"].unique().to_list())
        assert "warmup" not in statuses, f"Unexpected 'warmup' status in EWMA mode; found: {statuses}"


# ─── Signal quality (IC / ICIR) ──────────────────────────────────────────────


class TestSignalQuality:
    """IC and ICIR values are finite and within a reasonable range."""

    def test_ic_mean_is_finite(self, notebook_engine: BasanosEngine) -> None:
        """IC mean must be a finite float."""
        assert math.isfinite(notebook_engine.ic_mean)

    def test_ic_std_is_positive_finite(self, notebook_engine: BasanosEngine) -> None:
        """IC std must be a finite positive float."""
        assert math.isfinite(notebook_engine.ic_std)
        assert notebook_engine.ic_std > 0

    def test_icir_is_finite(self, notebook_engine: BasanosEngine) -> None:
        """ICIR must be a finite float."""
        assert math.isfinite(notebook_engine.icir)

    def test_rank_ic_mean_is_finite(self, notebook_engine: BasanosEngine) -> None:
        """Rank IC mean must be a finite float."""
        assert math.isfinite(notebook_engine.rank_ic_mean)

    def test_ic_mean_concrete(self, notebook_engine: BasanosEngine) -> None:
        """IC mean must match the deterministic expected value (fixed seed 2024)."""
        assert notebook_engine.ic_mean == pytest.approx(0.010676938245513712, rel=1e-4)

    def test_icir_concrete(self, notebook_engine: BasanosEngine) -> None:
        """ICIR must match the deterministic expected value (fixed seed 2024)."""
        assert notebook_engine.icir == pytest.approx(0.01920162208642282, rel=1e-4)


# ─── NAV ──────────────────────────────────────────────────────────────────────


class TestNAV:
    """Portfolio NAV is positive and deterministic (fixed seed, fixed config)."""

    def test_nav_row_count(self, notebook_portfolio: Any, notebook_prices: pl.DataFrame) -> None:
        """nav_accumulated must have one row per price timestamp."""
        assert notebook_portfolio.nav_accumulated.height == notebook_prices.height

    def test_final_nav_positive(self, notebook_portfolio: Any) -> None:
        """Final NAV must be strictly positive."""
        final = notebook_portfolio.nav_accumulated["NAV_accumulated"][-1]
        assert final > 0

    def test_final_nav_concrete(self, notebook_portfolio: Any) -> None:
        """Final NAV must match the deterministic expected value (fixed seed 2024)."""
        final = notebook_portfolio.nav_accumulated["NAV_accumulated"][-1]
        assert final == pytest.approx(28_515_151.968557302, rel=1e-4)

    def test_nav_starts_near_aum(self, notebook_portfolio: Any, notebook_cfg: BasanosConfig) -> None:
        """On day 1 there are no realised P&L, so NAV must be close to AUM."""
        first_nav = notebook_portfolio.nav_accumulated["NAV_accumulated"][0]
        assert abs(first_nav - notebook_cfg.aum) / notebook_cfg.aum < _INITIAL_NAV_TOLERANCE


# ─── Direct notebook execution ───────────────────────────────────────────────


def test_notebook_executes() -> None:
    """Execute end_to_end.py directly via marimo export html (no sandbox).

    This catches regressions in notebook cell code itself, not just the API
    that the mirror tests validate.
    """
    result = subprocess.run(  # nosec
        [sys.executable, "-m", "marimo", "export", "html", str(_NOTEBOOK), "-o", "/dev/null"],
        capture_output=True,
        text=True,
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    failure_keywords = ["cells failed to execute", "marimoexceptionraisederror"]
    for kw in failure_keywords:
        assert kw.lower() not in combined.lower(), (
            f"Notebook {_NOTEBOOK.name} reported cell failures (keyword '{kw}'):\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
    assert result.returncode == 0, (
        f"marimo export returned non-zero for {_NOTEBOOK.name}:\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
