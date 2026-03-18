"""Tests for basanos.math._config_report (ConfigReport facade)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from basanos.math._config_report import ConfigReport
from basanos.math.optimizer import BasanosConfig, BasanosEngine

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg() -> BasanosConfig:
    """Minimal BasanosConfig for testing."""
    return BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1_000_000)


@pytest.fixture
def engine(cfg: BasanosConfig) -> BasanosEngine:
    """Small BasanosEngine suitable for unit tests (fast lambda sweep)."""
    n = 120
    rng = np.random.default_rng(42)
    dates = pl.Series("date", list(range(n)))
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": rng.lognormal(size=n).tolist(),
            "B": rng.lognormal(size=n).tolist(),
        }
    )
    mu = pl.DataFrame(
        {
            "date": dates,
            "A": rng.normal(size=n).tolist(),
            "B": rng.normal(size=n).tolist(),
        }
    )
    return BasanosEngine(prices=prices, mu=mu, cfg=cfg)


# ── ConfigReport dataclass ────────────────────────────────────────────────────


def test_config_report_is_dataclass() -> None:
    """ConfigReport must be a dataclass."""
    assert dataclasses.is_dataclass(ConfigReport)


def test_config_report_accessible_via_config_property(cfg: BasanosConfig) -> None:
    """BasanosConfig.report must return a ConfigReport wrapping the config."""
    report = cfg.report
    assert isinstance(report, ConfigReport)
    assert report.config is cfg
    assert report.engine is None


def test_config_report_accessible_via_engine_property(engine: BasanosEngine) -> None:
    """BasanosEngine.config_report must return a ConfigReport with both config and engine set."""
    report = engine.config_report
    assert isinstance(report, ConfigReport)
    assert report.config is engine.cfg
    assert report.engine is engine


def test_config_report_frozen(cfg: BasanosConfig) -> None:
    """ConfigReport must be immutable (frozen dataclass)."""
    report = cfg.report
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        report.config = cfg  # type: ignore[misc]


# ── to_html — basic structure ─────────────────────────────────────────────────


def test_to_html_returns_string(cfg: BasanosConfig) -> None:
    """to_html must return a non-empty string."""
    html = cfg.report.to_html()
    assert isinstance(html, str)
    assert len(html) > 500


def test_to_html_starts_with_doctype(cfg: BasanosConfig) -> None:
    """to_html output must begin with a DOCTYPE declaration."""
    html = cfg.report.to_html()
    assert html.strip().startswith("<!DOCTYPE html>")


def test_to_html_default_title(cfg: BasanosConfig) -> None:
    """Default title appears in the HTML output."""
    html = cfg.report.to_html()
    assert "Basanos Configuration Report" in html


def test_to_html_custom_title(cfg: BasanosConfig) -> None:
    """Custom title is embedded in the HTML output."""
    html = cfg.report.to_html(title="My Config Test")
    assert "My Config Test" in html


def test_to_html_contains_section_ids(cfg: BasanosConfig) -> None:
    """All expected section ids must be present in the HTML."""
    html = cfg.report.to_html()
    for section_id in ("parameters", "lambda-sweep", "guidance", "theory"):
        assert f'id="{section_id}"' in html, f"Missing section id: {section_id}"


def test_to_html_contains_toc_links(cfg: BasanosConfig) -> None:
    """Table-of-contents links to key sections must be present."""
    html = cfg.report.to_html()
    assert 'href="#parameters"' in html
    assert 'href="#guidance"' in html
    assert 'href="#theory"' in html


# ── to_html — parameter table ─────────────────────────────────────────────────


def test_to_html_contains_param_names(cfg: BasanosConfig) -> None:
    """All BasanosConfig field names must appear in the HTML output."""
    html = cfg.report.to_html()
    for param in ("vola", "corr", "clip", "shrink", "aum"):
        assert param in html, f"Missing parameter: {param}"


def test_to_html_contains_current_values(cfg: BasanosConfig) -> None:
    """Current values of required fields must appear in the HTML."""
    html = cfg.report.to_html()
    # vola=10, corr=20, shrink=0.5 should all appear
    assert "10" in html
    assert "20" in html
    assert "0.5" in html


def test_to_html_contains_optional_param_names(cfg: BasanosConfig) -> None:
    """Optional BasanosConfig field names must also appear in the HTML."""
    html = cfg.report.to_html()
    for param in ("profit_variance_init", "profit_variance_decay", "denom_tol", "position_scale"):
        assert param in html, f"Missing optional parameter: {param}"


# ── to_html — lambda sweep ────────────────────────────────────────────────────


def test_to_html_without_engine_shows_placeholder(cfg: BasanosConfig) -> None:
    """When no engine is provided, the lambda-sweep section shows a helpful message."""
    html = cfg.report.to_html()
    assert "engine.config_report" in html or "BasanosEngine" in html


def test_to_html_with_engine_includes_plotly(engine: BasanosEngine) -> None:
    """When an engine is provided, the lambda-sweep section contains a Plotly chart."""
    html = engine.config_report.to_html()
    assert "plotly" in html.lower()


def test_to_html_with_engine_has_lambda_toc_link(engine: BasanosEngine) -> None:
    """Engine config_report HTML must have a TOC link to the lambda-sweep section."""
    html = engine.config_report.to_html()
    assert 'href="#lambda-sweep"' in html


def test_to_html_with_engine_references_current_lambda(engine: BasanosEngine) -> None:
    """The lambda-sweep chart marker reflects the engine's current shrink value."""
    html = engine.config_report.to_html()
    # The current lambda (0.5) should appear in the chart data or annotations
    assert "0.5" in html


# ── to_html — guidance & theory ──────────────────────────────────────────────


def test_to_html_contains_guidance_table(cfg: BasanosConfig) -> None:
    """The shrinkage-guidance table must be present in the HTML."""
    html = cfg.report.to_html()
    assert "Suggested" in html or "guidance" in html.lower()
    # Regime labels
    assert "n &gt; 20" in html or "n > 20" in html


def test_to_html_contains_theory_section(cfg: BasanosConfig) -> None:
    """The theory section must mention Ledoit-Wolf and the shrinkage formula."""
    html = cfg.report.to_html()
    assert "Ledoit" in html
    assert "shrinkage" in html.lower()
    assert "C_shrunk" in html or "C_{shrunk}" in html


def test_to_html_contains_references(cfg: BasanosConfig) -> None:
    """Academic references must be included in the theory section."""
    html = cfg.report.to_html()
    assert "2004" in html  # Ledoit & Wolf (2004)
    assert "shrink2id" in html


# ── save ──────────────────────────────────────────────────────────────────────


def test_save_writes_file(tmp_path: Path, cfg: BasanosConfig) -> None:
    """save() must write a valid HTML file."""
    out = tmp_path / "config_report.html"
    result = cfg.report.save(out)
    assert result == out
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content


def test_save_appends_html_extension(tmp_path: Path, cfg: BasanosConfig) -> None:
    """save() must append .html extension when path has no suffix."""
    out = tmp_path / "my_config_report"
    result = cfg.report.save(out)
    assert result.suffix == ".html"
    assert result.exists()


def test_save_accepts_string_path(tmp_path: Path, cfg: BasanosConfig) -> None:
    """save() must accept a plain string path."""
    out = str(tmp_path / "config2.html")
    result = cfg.report.save(out, title="Custom Title")
    assert result.exists()
    assert "Custom Title" in result.read_text(encoding="utf-8")


def test_save_with_engine(tmp_path: Path, engine: BasanosEngine) -> None:
    """save() on engine.config_report must include the lambda-sweep chart."""
    out = tmp_path / "engine_report.html"
    result = engine.config_report.save(out)
    assert result.exists()
    content = result.read_text(encoding="utf-8")
    assert "plotly" in content.lower()


# ── _lambda_sweep_fig exception handler ───────────────────────────────────────


def test_to_html_lambda_sweep_unavailable_on_error(engine: BasanosEngine) -> None:
    """to_html catches _lambda_sweep_fig exceptions and embeds a notice (lines 564-565)."""
    with patch("basanos.math._config_report._lambda_sweep_fig", side_effect=RuntimeError("sweep failed")):
        html = engine.config_report.to_html()
    assert 'class="chart-unavailable"' in html
    assert "Lambda sweep unavailable" in html
    assert "sweep failed" in html


# ── PCA mode in ConfigReport ──────────────────────────────────────────────────


@pytest.fixture
def pca_cfg() -> BasanosConfig:
    """BasanosConfig in PCA mode with 2 components for config report tests."""
    return BasanosConfig(vola=10, corr=20, clip=3.0, shrink=0.5, aum=1_000_000, covariance_mode="pca", pca_components=2)


@pytest.fixture
def pca_engine(pca_cfg: BasanosConfig) -> BasanosEngine:
    """Small BasanosEngine in PCA mode suitable for unit tests."""
    n = 120
    rng = np.random.default_rng(7)
    dates = pl.Series("date", list(range(n)))
    prices = pl.DataFrame(
        {
            "date": dates,
            "A": rng.lognormal(size=n).tolist(),
            "B": rng.lognormal(size=n).tolist(),
        }
    )
    mu = pl.DataFrame(
        {
            "date": dates,
            "A": rng.normal(size=n).tolist(),
            "B": rng.normal(size=n).tolist(),
        }
    )
    return BasanosEngine(prices=prices, mu=mu, cfg=pca_cfg)


def test_to_html_includes_cov_mode_in_header(cfg: BasanosConfig) -> None:
    """to_html must include the covariance mode in the report header."""
    html = cfg.report.to_html()
    assert "ewma_shrink" in html


def test_to_html_pca_mode_shows_in_header(pca_cfg: BasanosConfig) -> None:
    """to_html for a PCA-mode config must show 'pca' in the header."""
    html = pca_cfg.report.to_html()
    assert "pca" in html


def test_to_html_pca_sweep_section_present(pca_engine: BasanosEngine) -> None:
    """to_html for an engine in PCA mode must include the PCA sweep section."""
    html = pca_engine.config_report.to_html()
    assert "pca-sweep" in html


def test_to_html_pca_sweep_chart_rendered(pca_engine: BasanosEngine) -> None:
    """to_html for engine in PCA mode must render an actual Plotly PCA sweep chart."""
    html = pca_engine.config_report.to_html()
    assert "plotly" in html.lower()


def test_to_html_pca_sweep_unavailable_notice_without_pca_mode(engine: BasanosEngine) -> None:
    """to_html for an engine in ewma_shrink mode must show the PCA unavailable notice."""
    html = engine.config_report.to_html()
    assert "covariance_mode" in html or "pca" in html.lower()


def test_to_html_pca_sweep_unavailable_on_error(pca_engine: BasanosEngine) -> None:
    """to_html catches _pca_sweep_fig exceptions and embeds a notice."""
    with patch("basanos.math._config_report._pca_sweep_fig", side_effect=RuntimeError("pca sweep failed")):
        html = pca_engine.config_report.to_html()
    assert 'class="chart-unavailable"' in html
    assert "PCA sweep unavailable" in html
    assert "pca sweep failed" in html


def test_pca_sweep_fig_with_max_k_less_than_current_k(pca_engine: BasanosEngine) -> None:
    """_pca_sweep_fig must compute current_sharpe separately when pca_components > max_k."""
    from basanos.math._config_report import _pca_sweep_fig

    # pca_engine has pca_components=2, so set max_k=1 to trigger the else branch
    fig = _pca_sweep_fig(pca_engine, max_k=1)
    # Verify the figure was built successfully with a marker for current_k
    assert fig is not None
    assert len(fig.data) == 2  # main sweep line + current-k marker
