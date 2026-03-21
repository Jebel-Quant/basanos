"""CI execution gate for the diagnostics notebook.

This module drives ``book/marimo/notebooks/diagnostics.py`` directly via
``app.run()`` so that any change to the notebook's data-generation cells,
configuration, or engine construction is automatically reflected here.
No constants or logic are duplicated from the notebook.

Covered diagnostic properties (both EWMA-shrink and Sliding Window modes):

- ``position_status``   — per-row label: warmup / zero_signal / degenerate / valid
- ``condition_number``  — condition number κ of the correlation matrix
- ``effective_rank``    — entropy-based effective rank
- ``solver_residual``   — Euclidean residual ‖C·x − μ‖₂
- ``signal_utilisation``— fraction of μᵢ surviving the correlation filter

The SW streaming path (``SlidingWindowConfig``) was added to the public API
after the notebook was first written, making this gate especially important.
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from basanos.math import BasanosEngine

_NOTEBOOK = Path(__file__).parents[2] / "book/marimo/notebooks/diagnostics.py"


# ─── Notebook execution fixture ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def notebook_defs() -> Mapping[str, Any]:
    """Run the diagnostics notebook once via app.run() and return all cell definitions.

    This is the single source of truth for all fixtures in this module.
    Any change to the notebook (data generation, config, engine construction)
    is automatically picked up here — no mirrored constants to drift.
    """
    sys.path.insert(0, str(_NOTEBOOK.parent))
    try:
        from diagnostics import app  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    _outputs, defs = app.run()
    return defs


@pytest.fixture(scope="module")
def notebook_prices(notebook_defs: Mapping[str, Any]) -> pl.DataFrame:
    """Prices DataFrame as produced by cell_04 of the diagnostics notebook."""
    return notebook_defs["prices"]


@pytest.fixture(scope="module")
def ewma_engine(notebook_defs: Mapping[str, Any]) -> BasanosEngine:
    """BasanosEngine in EWMA-shrink mode as constructed by cell_07 of the notebook."""
    return notebook_defs["engine"]


@pytest.fixture(scope="module")
def sw_engine(notebook_defs: Mapping[str, Any]) -> BasanosEngine:
    """BasanosEngine in Sliding Window mode as constructed by the SW cell of the notebook."""
    return notebook_defs["sw_engine"]


# ─── position_status ─────────────────────────────────────────────────────────


class TestPositionStatusEwma:
    """position_status output matches the notebook's expected schema (EWMA mode).

    Note: EWMA mode does **not** produce a ``'warmup'`` status code.
    Rows before the EWMA estimator converges are labelled ``'degenerate'``
    (the NaN correlation matrix causes the linear solve to fail).
    Only sliding-window mode explicitly emits ``'warmup'``.
    """

    def test_columns(self, ewma_engine: BasanosEngine) -> None:
        """position_status must have exactly ['date', 'status'] columns."""
        ps = ewma_engine.position_status
        assert ps.columns == ["date", "status"]

    def test_row_count(self, ewma_engine: BasanosEngine, notebook_prices: pl.DataFrame) -> None:
        """position_status must have one row per price timestamp."""
        assert ewma_engine.position_status.height == notebook_prices.height

    def test_valid_status_codes(self, ewma_engine: BasanosEngine) -> None:
        """Every status value must be one of the four defined codes."""
        valid = {"warmup", "zero_signal", "degenerate", "valid"}
        actual = set(ewma_engine.position_status["status"].unique().to_list())
        assert actual.issubset(valid), f"Unexpected status codes: {actual - valid}"

    def test_ewma_mode_never_emits_warmup(self, ewma_engine: BasanosEngine) -> None:
        """EWMA mode must not emit the 'warmup' code; that code is SW-only.

        During EWMA estimation, rows with an unconverged (NaN) correlation
        matrix fail the linear solve and are labelled 'degenerate', not 'warmup'.
        """
        statuses = set(ewma_engine.position_status["status"].unique().to_list())
        assert "warmup" not in statuses, f"EWMA mode unexpectedly emitted 'warmup' status; found codes: {statuses}"

    def test_early_rows_are_not_valid_with_low_shrinkage(self, ewma_engine: BasanosEngine) -> None:
        """With shrink=0.99, the NaN EWMA matrix makes early rows degenerate, not valid."""
        # The notebook uses shrink=0.99 to deliberately surface ill-conditioning
        early_statuses = set(ewma_engine.position_status.head(30)["status"].to_list())
        assert "valid" not in early_statuses, f"Unexpected 'valid' rows in early EWMA period; found: {early_statuses}"

    def test_zero_signal_window_produces_zero_signal(self, ewma_engine: BasanosEngine) -> None:
        """Rows with zero mu must not produce any 'valid' status.

        The notebook injects a zero-signal window; we identify those rows from
        position_status itself (zero_signal or degenerate, never valid).
        """
        statuses = ewma_engine.position_status["status"].to_list()
        zero_signal_indices = [i for i, s in enumerate(statuses) if s == "zero_signal"]
        assert len(zero_signal_indices) > 0, "Expected at least one zero_signal row in EWMA engine"
        # zero_signal rows must never appear as valid
        for i in zero_signal_indices:
            assert statuses[i] != "valid", f"Row {i} has zero_signal mu but status 'valid'"

    def test_has_valid_rows_after_ewma_converges(self, ewma_engine: BasanosEngine) -> None:
        """After the EWMA estimator converges there must be at least one 'valid' row."""
        # Use the tail: by row 300+ the EWMA should have converged
        tail_statuses = ewma_engine.position_status.tail(200)["status"].to_list()
        assert "valid" in tail_statuses


class TestPositionStatusSw:
    """position_status output matches the expected schema (Sliding Window mode)."""

    def test_columns(self, sw_engine: BasanosEngine) -> None:
        """position_status must have exactly ['date', 'status'] columns."""
        assert sw_engine.position_status.columns == ["date", "status"]

    def test_row_count(self, sw_engine: BasanosEngine, notebook_prices: pl.DataFrame) -> None:
        """position_status must have one row per price timestamp."""
        assert sw_engine.position_status.height == notebook_prices.height

    def test_valid_status_codes(self, sw_engine: BasanosEngine) -> None:
        """Every status value must be one of the four defined codes."""
        valid = {"warmup", "zero_signal", "degenerate", "valid"}
        actual = set(sw_engine.position_status["status"].unique().to_list())
        assert actual.issubset(valid), f"Unexpected status codes: {actual - valid}"

    def test_warmup_rows_are_warmup(self, sw_engine: BasanosEngine) -> None:
        """First window-1 rows must be in warmup for SW mode."""
        # SlidingWindowConfig(window=80) -> first 79 rows are warmup
        expected_warmup = 79
        statuses = sw_engine.position_status.head(expected_warmup)["status"].to_list()
        assert all(s == "warmup" for s in statuses)

    def test_has_valid_rows_after_warmup(self, sw_engine: BasanosEngine) -> None:
        """After the sliding window fills, at least some rows must be 'valid'."""
        tail_statuses = sw_engine.position_status.tail(200)["status"].to_list()
        assert "valid" in tail_statuses


# ─── condition_number ─────────────────────────────────────────────────────────


class TestConditionNumberEwma:
    """condition_number diagnostic (EWMA mode)."""

    def test_columns(self, ewma_engine: BasanosEngine) -> None:
        """condition_number must have exactly ['date', 'condition_number'] columns."""
        assert ewma_engine.condition_number.columns == ["date", "condition_number"]

    def test_row_count(self, ewma_engine: BasanosEngine, notebook_prices: pl.DataFrame) -> None:
        """condition_number must have one row per price timestamp."""
        assert ewma_engine.condition_number.height == notebook_prices.height

    def test_warmup_is_nan(self, ewma_engine: BasanosEngine) -> None:
        """Condition numbers during warmup must be NaN."""
        warmup_n = ewma_engine.cfg.corr
        kappas = ewma_engine.condition_number.head(warmup_n)["condition_number"].to_list()
        assert all(v is None or (isinstance(v, float) and math.isnan(v)) for v in kappas)

    def test_post_warmup_has_finite_positive_values(self, ewma_engine: BasanosEngine) -> None:
        """Condition numbers after warmup must be finite and ≥ 1."""
        warmup_n = ewma_engine.cfg.corr
        vals = ewma_engine.condition_number.slice(warmup_n)["condition_number"].drop_nulls().to_list()
        assert len(vals) > 0, "No finite condition numbers found after warmup"
        assert all(v >= 1.0 - 1e-9 for v in vals), "condition_number < 1 found"


class TestConditionNumberSw:
    """condition_number diagnostic (Sliding Window mode)."""

    def test_columns(self, sw_engine: BasanosEngine) -> None:
        """condition_number must have exactly ['date', 'condition_number'] columns."""
        assert sw_engine.condition_number.columns == ["date", "condition_number"]

    def test_row_count(self, sw_engine: BasanosEngine, notebook_prices: pl.DataFrame) -> None:
        """condition_number must have one row per price timestamp."""
        assert sw_engine.condition_number.height == notebook_prices.height

    def test_warmup_is_nan(self, sw_engine: BasanosEngine) -> None:
        """Condition numbers during the SW warmup window must be NaN."""
        expected_warmup = 79
        kappas = sw_engine.condition_number.head(expected_warmup)["condition_number"].to_list()
        assert all(v is None or (isinstance(v, float) and math.isnan(v)) for v in kappas)

    def test_post_warmup_has_finite_values(self, sw_engine: BasanosEngine) -> None:
        """Some post-warmup condition numbers must be finite and positive."""
        vals = sw_engine.condition_number.tail(200)["condition_number"].to_list()
        finite = [v for v in vals if v is not None and isinstance(v, float) and np.isfinite(v)]
        assert len(finite) > 0, "No finite condition numbers found in SW mode"


# ─── solver_residual ─────────────────────────────────────────────────────────


class TestSolverResidualEwma:
    """solver_residual diagnostic (EWMA mode)."""

    def test_columns(self, ewma_engine: BasanosEngine) -> None:
        """solver_residual must have exactly ['date', 'residual'] columns."""
        assert ewma_engine.solver_residual.columns == ["date", "residual"]

    def test_row_count(self, ewma_engine: BasanosEngine, notebook_prices: pl.DataFrame) -> None:
        """solver_residual must have one row per price timestamp."""
        assert ewma_engine.solver_residual.height == notebook_prices.height

    def test_residuals_non_negative_where_finite(self, ewma_engine: BasanosEngine) -> None:
        """Residuals must be >= 0 wherever they are finite."""
        residuals = ewma_engine.solver_residual["residual"].drop_nulls().to_list()
        finite = [v for v in residuals if isinstance(v, float) and np.isfinite(v)]
        assert all(v >= 0.0 for v in finite), "Negative solver residual found"

    def test_zero_signal_window_has_zero_residual(self, ewma_engine: BasanosEngine) -> None:
        """When mu = 0, no solve is performed and residual must be 0 or None."""
        ps = ewma_engine.position_status["status"].to_list()
        res = ewma_engine.solver_residual["residual"].to_list()
        for status, r in zip(ps, res, strict=True):
            if status == "zero_signal":
                assert r == 0.0 or r is None, f"Expected 0 residual for zero_signal row, got {r}"


class TestSolverResidualSw:
    """solver_residual diagnostic (Sliding Window mode)."""

    def test_columns(self, sw_engine: BasanosEngine) -> None:
        """solver_residual must have exactly ['date', 'residual'] columns."""
        assert sw_engine.solver_residual.columns == ["date", "residual"]

    def test_row_count(self, sw_engine: BasanosEngine, notebook_prices: pl.DataFrame) -> None:
        """solver_residual must have one row per price timestamp."""
        assert sw_engine.solver_residual.height == notebook_prices.height

    def test_residuals_non_negative_where_finite(self, sw_engine: BasanosEngine) -> None:
        """Residuals must be >= 0 wherever they are finite."""
        residuals = sw_engine.solver_residual["residual"].drop_nulls().to_list()
        finite = [v for v in residuals if isinstance(v, float) and np.isfinite(v)]
        assert all(v >= 0.0 for v in finite), "Negative solver residual found in SW mode"


# ─── effective_rank ───────────────────────────────────────────────────────────


class TestEffectiveRankEwma:
    """effective_rank diagnostic (EWMA mode)."""

    def test_columns(self, ewma_engine: BasanosEngine) -> None:
        """effective_rank must have exactly ['date', 'effective_rank'] columns."""
        assert ewma_engine.effective_rank.columns == ["date", "effective_rank"]

    def test_row_count(self, ewma_engine: BasanosEngine, notebook_prices: pl.DataFrame) -> None:
        """effective_rank must have one row per price timestamp."""
        assert ewma_engine.effective_rank.height == notebook_prices.height

    def test_post_warmup_in_valid_range(self, ewma_engine: BasanosEngine) -> None:
        """Effective rank must be in [1, n_assets] after warmup."""
        n_assets = len(ewma_engine.prices.columns) - 1  # exclude date column
        warmup_n = ewma_engine.cfg.corr
        vals = ewma_engine.effective_rank.slice(warmup_n)["effective_rank"].drop_nulls().to_list()
        for v in vals:
            if isinstance(v, float) and np.isfinite(v):
                assert 1.0 - 1e-9 <= v <= n_assets + 1e-9, f"effective_rank out of range: {v}"


class TestEffectiveRankSw:
    """effective_rank diagnostic (Sliding Window mode)."""

    def test_columns(self, sw_engine: BasanosEngine) -> None:
        """effective_rank must have exactly ['date', 'effective_rank'] columns."""
        assert sw_engine.effective_rank.columns == ["date", "effective_rank"]

    def test_row_count(self, sw_engine: BasanosEngine, notebook_prices: pl.DataFrame) -> None:
        """effective_rank must have one row per price timestamp."""
        assert sw_engine.effective_rank.height == notebook_prices.height


# ─── signal_utilisation ───────────────────────────────────────────────────────


class TestSignalUtilisationEwma:
    """signal_utilisation diagnostic (EWMA mode)."""

    def test_columns_match_prices(self, ewma_engine: BasanosEngine) -> None:
        """signal_utilisation must have the same columns as prices (date + assets)."""
        assert ewma_engine.signal_utilisation.columns == ewma_engine.prices.columns

    def test_row_count(self, ewma_engine: BasanosEngine, notebook_prices: pl.DataFrame) -> None:
        """signal_utilisation must have one row per price timestamp."""
        assert ewma_engine.signal_utilisation.height == notebook_prices.height


class TestSignalUtilisationSw:
    """signal_utilisation diagnostic (Sliding Window mode)."""

    def test_columns_match_prices(self, sw_engine: BasanosEngine) -> None:
        """signal_utilisation must have the same columns as prices (date + assets)."""
        assert sw_engine.signal_utilisation.columns == sw_engine.prices.columns

    def test_row_count(self, sw_engine: BasanosEngine, notebook_prices: pl.DataFrame) -> None:
        """signal_utilisation must have one row per price timestamp."""
        assert sw_engine.signal_utilisation.height == notebook_prices.height


# ─── Direct notebook execution ───────────────────────────────────────────────


def test_notebook_executes() -> None:
    """Execute diagnostics.py directly via marimo export html (no sandbox).

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
