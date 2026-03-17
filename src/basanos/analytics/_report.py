"""HTML report generation for portfolio analytics.

This module defines the Report facade which produces a self-contained HTML
document containing all relevant performance numbers and interactive Plotly
visualisations for a Portfolio.

Examples:
    >>> import dataclasses
    >>> from basanos.analytics._report import Report
    >>> dataclasses.is_dataclass(Report)
    True
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl

if TYPE_CHECKING:
    from .portfolio import Portfolio


# ── Formatting helpers ────────────────────────────────────────────────────────


def _is_finite(v: object) -> TypeGuard[int | float]:
    """Return True when *v* is a real, finite number."""
    if not isinstance(v, (int, float)):
        return False
    return math.isfinite(float(v))


def _fmt(value: object, fmt: str = ".4f", suffix: str = "") -> str:
    """Format *value* for display in an HTML table cell.

    Returns ``"N/A"`` for ``None``, ``NaN``, or non-finite values.
    """
    if not _is_finite(value):
        return "N/A"
    return f"{float(value):{fmt}}{suffix}"


# ── Stats table ───────────────────────────────────────────────────────────────

_METRIC_FORMATS: dict[str, tuple[str, str]] = {
    "avg_return": (".6f", ""),
    "avg_win": (".6f", ""),
    "avg_loss": (".6f", ""),
    "best": (".6f", ""),
    "worst": (".6f", ""),
    "sharpe": (".2f", ""),
    "calmar": (".2f", ""),
    "recovery_factor": (".2f", ""),
    "max_drawdown": (".2%", ""),
    "avg_drawdown": (".2%", ""),
    "max_drawdown_duration": (".0f", " days"),
    "win_rate": (".1%", ""),
    "monthly_win_rate": (".1%", ""),
    "profit_factor": (".2f", ""),
    "payoff_ratio": (".2f", ""),
    "volatility": (".2%", ""),
    "skew": (".2f", ""),
    "kurtosis": (".2f", ""),
    "value_at_risk": (".6f", ""),
    "conditional_value_at_risk": (".6f", ""),
}

_METRIC_LABELS: dict[str, str] = {
    "avg_return": "Avg Return",
    "avg_win": "Avg Win",
    "avg_loss": "Avg Loss",
    "best": "Best Period",
    "worst": "Worst Period",
    "sharpe": "Sharpe Ratio",
    "calmar": "Calmar Ratio",
    "recovery_factor": "Recovery Factor",
    "max_drawdown": "Max Drawdown",
    "avg_drawdown": "Avg Drawdown",
    "max_drawdown_duration": "Max DD Duration",
    "win_rate": "Win Rate",
    "monthly_win_rate": "Monthly Win Rate",
    "profit_factor": "Profit Factor",
    "payoff_ratio": "Payoff Ratio",
    "volatility": "Volatility (ann.)",
    "skew": "Skewness",
    "kurtosis": "Kurtosis",
    "value_at_risk": "VaR (95 %)",
    "conditional_value_at_risk": "CVaR (95 %)",
}

# Metrics where the *highest* value across assets is highlighted.
_HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {"sharpe", "calmar", "recovery_factor", "win_rate", "monthly_win_rate", "profit_factor", "payoff_ratio"}
)

_CATEGORIES: list[tuple[str, list[str]]] = [
    ("Returns", ["avg_return", "avg_win", "avg_loss", "best", "worst"]),
    ("Risk-Adjusted Performance", ["sharpe", "calmar", "recovery_factor"]),
    ("Drawdown", ["max_drawdown", "avg_drawdown", "max_drawdown_duration"]),
    ("Win / Loss", ["win_rate", "monthly_win_rate", "profit_factor", "payoff_ratio"]),
    ("Distribution & Risk", ["volatility", "skew", "kurtosis", "value_at_risk", "conditional_value_at_risk"]),
]


def _stats_table_html(summary: pl.DataFrame) -> str:
    """Render a stats summary DataFrame as a styled HTML table.

    Args:
        summary: Output of :py:meth:`Stats.summary` — one row per metric,
            one column per asset plus a ``metric`` column.

    Returns:
        An HTML ``<table>`` string ready to embed in a page.
    """
    assets = [c for c in summary.columns if c != "metric"]

    # Build a fast lookup: metric_name → {asset: value}
    metric_data: dict[str, dict[str, object]] = {}
    for row in summary.iter_rows(named=True):
        name = str(row["metric"])
        metric_data[name] = {a: row.get(a) for a in assets}

    header_cells = "".join(f'<th class="asset-header">{a}</th>' for a in assets)
    rows_html_parts: list[str] = []

    for category_label, metrics in _CATEGORIES:
        rows_html_parts.append(
            f'<tr class="table-section-header">'
            f'<td colspan="{len(assets) + 1}"><strong>{category_label}</strong></td>'
            f"</tr>\n"
        )
        for metric in metrics:
            if metric not in metric_data:
                continue
            fmt, suffix = _METRIC_FORMATS.get(metric, (".4f", ""))
            label = _METRIC_LABELS.get(metric, metric.replace("_", " ").title())
            values = metric_data[metric]

            # Find the best asset to highlight (only for higher-is-better metrics)
            best_asset: str | None = None
            if metric in _HIGHER_IS_BETTER:
                finite_pairs = [(a, float(v)) for a, v in values.items() if _is_finite(v)]
                if finite_pairs:
                    best_asset = max(finite_pairs, key=lambda x: x[1])[0]

            cells = "".join(
                f'<td class="metric-value{"  best-value" if a == best_asset else ""}">'
                f"{_fmt(values.get(a), fmt, suffix)}</td>"
                for a in assets
            )
            rows_html_parts.append(f'<tr><td class="metric-name">{label}</td>{cells}</tr>\n')

    rows_html = "".join(rows_html_parts)
    return (
        '<table class="stats-table">'
        "<thead><tr>"
        f'<th class="metric-header">Metric</th>{header_cells}'
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
    )


# ── CSS / HTML templates ──────────────────────────────────────────────────────

_CSS = """
/* ── Reset & Base ─────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    line-height: 1.6;
}

/* ── Header ───────────────────────────────────────── */
.report-header {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 100%);
    border-bottom: 2px solid #2d3748;
    padding: 2.5rem 2rem 2rem;
}
.report-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #63b3ed;
    letter-spacing: -0.5px;
}
.report-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    margin-top: 0.75rem;
    font-size: 0.875rem;
    color: #a0aec0;
}
.report-meta span strong { color: #e2e8f0; }

/* ── Table of Contents ────────────────────────────── */
.toc {
    background: #1a1f35;
    border-bottom: 1px solid #2d3748;
    padding: 0.75rem 2rem;
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    font-size: 0.8rem;
    position: sticky;
    top: 0;
    z-index: 100;
}
.toc a {
    color: #63b3ed;
    text-decoration: none;
    opacity: 0.8;
    transition: opacity 0.2s;
}
.toc a:hover { opacity: 1; text-decoration: underline; }

/* ── Main Content ─────────────────────────────────── */
.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
}

/* ── Sections ─────────────────────────────────────── */
.section { margin-bottom: 3rem; }
.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #90cdf4;
    margin-bottom: 1.25rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2d3748;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::before {
    content: "";
    display: inline-block;
    width: 4px;
    height: 1.2em;
    background: #4299e1;
    border-radius: 2px;
}

/* ── Chart Grid ───────────────────────────────────── */
.chart-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
}
.chart-grid .chart-card.full-width { grid-column: 1 / -1; }
.chart-card {
    background: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1rem;
    overflow: hidden;
}
.chart-card .js-plotly-plot,
.chart-card .plotly-graph-div { width: 100% !important; }

/* ── Stats Table ──────────────────────────────────── */
.stats-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}
.stats-table th {
    background: #2d3748;
    color: #90cdf4;
    padding: 0.6rem 1rem;
    text-align: right;
    font-weight: 600;
    white-space: nowrap;
}
.stats-table th.metric-header { text-align: left; }
.stats-table th.asset-header  { text-align: right; }
.stats-table td {
    padding: 0.45rem 1rem;
    border-bottom: 1px solid #2d3748;
    text-align: right;
}
.stats-table td.metric-name {
    text-align: left;
    color: #cbd5e0;
    padding-left: 1.5rem;
}
.stats-table tr.table-section-header td {
    background: #1e2a3a;
    color: #4299e1;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.4rem 1rem;
    text-align: left;
}
.stats-table tbody tr:hover { background: #1e2a3a; }
.stats-table td.best-value { color: #68d391; font-weight: 600; }
.stats-table td.metric-value {
    font-family: "SFMono-Regular", Consolas, monospace;
}

/* ── Footer ───────────────────────────────────────── */
.report-footer {
    text-align: center;
    padding: 1.5rem;
    color: #4a5568;
    font-size: 0.75rem;
    border-top: 1px solid #2d3748;
    margin-top: 3rem;
}

@media (max-width: 900px) {
    .chart-grid { grid-template-columns: 1fr; }
    .chart-card.full-width { grid-column: 1; }
}
"""


# ── Report dataclass ──────────────────────────────────────────────────────────


def _figure_div(fig: go.Figure, include_plotlyjs: bool | str) -> str:
    """Return an HTML div string for *fig*.

    Args:
        fig: Plotly figure to serialise.
        include_plotlyjs: Passed directly to :func:`plotly.io.to_html`.
            Pass ``"cdn"`` for the first figure so the CDN script tag is
            injected; pass ``False`` for all subsequent figures.

    Returns:
        HTML string (not a full page).
    """
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
    )


@dataclasses.dataclass(frozen=True)
class Report:
    """Facade for generating HTML reports from a Portfolio.

    Provides a :py:meth:`to_html` method that assembles a self-contained,
    dark-themed HTML document with a performance-statistics table and
    multiple interactive Plotly charts.

    Usage::

        report = portfolio.report
        html_str = report.to_html()
        report.save("output/report.html")
    """

    portfolio: Portfolio

    def to_html(self, title: str = "Basanos Portfolio Report") -> str:
        """Render a full HTML report as a string.

        The document is self-contained: Plotly.js is loaded once from the
        CDN and all charts are embedded as ``<div>`` elements.  No external
        CSS framework is required.

        Args:
            title: HTML ``<title>`` text and visible page heading.

        Returns:
            A complete HTML document as a :class:`str`.
        """
        pf = self.portfolio

        # ── Metadata ──────────────────────────────────────────────────────────
        has_date = "date" in pf.prices.columns
        if has_date:
            dates = pf.prices["date"]
            start_date = str(dates.min())
            end_date = str(dates.max())
            n_periods = pf.prices.height
            period_info = f"{start_date} → {end_date} &nbsp;|&nbsp; {n_periods:,} periods"
        else:
            start_date = ""
            end_date = ""
            period_info = f"{pf.prices.height:,} periods"

        assets_list = ", ".join(pf.assets)

        # ── Figures ───────────────────────────────────────────────────────────
        # The first chart includes Plotly.js from CDN; subsequent ones reuse it.
        _first = True

        def _div(fig: go.Figure) -> str:
            nonlocal _first
            include = "cdn" if _first else False
            _first = False
            return _figure_div(fig, include)

        def _try_div(build_fig: object) -> str:
            """Call *build_fig()* and return the chart div; on error return a notice."""
            try:
                fig = build_fig()  # type: ignore[operator]
                return _div(fig)
            except Exception as exc:
                return f'<p class="chart-unavailable">Chart unavailable: {exc}</p>'

        snapshot_div = _try_div(pf.plots.snapshot)
        rolling_sharpe_div = _try_div(pf.plots.rolling_sharpe_plot)
        rolling_vol_div = _try_div(pf.plots.rolling_volatility_plot)
        annual_sharpe_div = _try_div(pf.plots.annual_sharpe_plot)
        monthly_heatmap_div = _try_div(pf.plots.monthly_returns_heatmap)
        corr_div = _try_div(pf.plots.correlation_heatmap)
        lead_lag_div = _try_div(pf.plots.lead_lag_ir_plot)
        trading_cost_div = _try_div(pf.plots.trading_cost_impact_plot)

        # ── Stats table ───────────────────────────────────────────────────────
        stats_table = _stats_table_html(pf.stats.summary())

        # ── Turnover table ────────────────────────────────────────────────────
        try:
            turnover_df = pf.turnover_summary()
            turnover_rows = "".join(
                f'<tr><td class="metric-name">{row["metric"].replace("_", " ").title()}</td>'
                f'<td class="metric-value">{row["value"]:.4f}</td></tr>'
                for row in turnover_df.iter_rows(named=True)
            )
            turnover_html = (
                '<table class="stats-table">'
                "<thead><tr>"
                '<th class="metric-header">Metric</th>'
                '<th class="asset-header">Value</th>'
                "</tr></thead>"
                f"<tbody>{turnover_rows}</tbody>"
                "</table>"
            )
        except Exception as exc:
            turnover_html = f'<p class="chart-unavailable">Turnover data unavailable: {exc}</p>'

        # ── Assemble HTML ─────────────────────────────────────────────────────
        footer_date = end_date if has_date else ""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>{_CSS}</style>
</head>
<body>

<header class="report-header">
  <h1>&#x1F4CA; {title}</h1>
  <div class="report-meta">
    <span><strong>Period:</strong> {period_info}</span>
    <span><strong>Assets:</strong> {assets_list}</span>
    <span><strong>AUM:</strong> {pf.aum:,.0f}</span>
  </div>
</header>

<nav class="toc">
  <a href="#performance">Performance</a>
  <a href="#risk">Risk</a>
  <a href="#annual">Annual</a>
  <a href="#monthly">Monthly Returns</a>
  <a href="#stats-table">Statistics</a>
  <a href="#correlation">Correlation</a>
  <a href="#leadlag">Lead / Lag</a>
  <a href="#costs">Trading Costs</a>
  <a href="#turnover">Turnover</a>
</nav>

<div class="container">

  <section class="section" id="performance">
    <h2 class="section-title">Portfolio Performance</h2>
    <div class="chart-card">{snapshot_div}</div>
  </section>

  <section class="section" id="risk">
    <h2 class="section-title">Risk Analysis</h2>
    <div class="chart-grid">
      <div class="chart-card">{rolling_sharpe_div}</div>
      <div class="chart-card">{rolling_vol_div}</div>
    </div>
  </section>

  <section class="section" id="annual">
    <h2 class="section-title">Annual Breakdown</h2>
    <div class="chart-card">{annual_sharpe_div}</div>
  </section>

  <section class="section" id="monthly">
    <h2 class="section-title">Monthly Returns</h2>
    <div class="chart-card">{monthly_heatmap_div}</div>
  </section>

  <section class="section" id="stats-table">
    <h2 class="section-title">Performance Statistics</h2>
    <div class="chart-card" style="overflow-x: auto;">{stats_table}</div>
  </section>

  <section class="section" id="correlation">
    <h2 class="section-title">Correlation Analysis</h2>
    <div class="chart-card">{corr_div}</div>
  </section>

  <section class="section" id="leadlag">
    <h2 class="section-title">Lead / Lag Information Ratio</h2>
    <div class="chart-card">{lead_lag_div}</div>
  </section>

  <section class="section" id="costs">
    <h2 class="section-title">Trading Cost Impact</h2>
    <div class="chart-card">{trading_cost_div}</div>
  </section>

  <section class="section" id="turnover">
    <h2 class="section-title">Turnover Summary</h2>
    <div class="chart-card" style="overflow-x: auto;">{turnover_html}</div>
  </section>

</div>

<footer class="report-footer">
  Generated by <strong>basanos</strong>&nbsp;|&nbsp;{footer_date}
</footer>

</body>
</html>"""

    def save(self, path: str | Path, title: str = "Basanos Portfolio Report") -> Path:
        """Save the HTML report to a file.

        A ``.html`` suffix is appended automatically when *path* has no
        file extension.

        Args:
            path: Destination file path.
            title: HTML ``<title>`` text and visible page heading.

        Returns:
            The resolved :class:`pathlib.Path` of the written file.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".html")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_html(title=title), encoding="utf-8")
        return p
