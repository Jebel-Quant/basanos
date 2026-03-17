"""HTML report generation for BasanosConfig parameter analysis.

This module defines the :class:`ConfigReport` facade which produces a
self-contained HTML document summarising all configuration parameters,
their constraints and descriptions, an interactive lambda-sweep chart
(when a :class:`~basanos.math.optimizer.BasanosEngine` is provided), a
shrinkage-guidance table, and a theory section on Ledoit-Wolf shrinkage.

Examples:
    >>> import dataclasses
    >>> from basanos.math._config_report import ConfigReport
    >>> dataclasses.is_dataclass(ConfigReport)
    True
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

if TYPE_CHECKING:
    from .optimizer import BasanosConfig, BasanosEngine


# ── CSS (reuses the same dark-theme palette as _report.py) ───────────────────

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
    max-width: 1200px;
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

/* ── Chart Card ───────────────────────────────────── */
.chart-card {
    background: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1rem;
    overflow: hidden;
}
.chart-card .js-plotly-plot,
.chart-card .plotly-graph-div { width: 100% !important; }
.chart-unavailable {
    color: #a0aec0;
    font-style: italic;
    padding: 1rem;
}

/* ── Parameter & Guidance Tables ─────────────────── */
.param-table, .guidance-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}
.param-table th, .guidance-table th {
    background: #2d3748;
    color: #90cdf4;
    padding: 0.6rem 1rem;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
}
.param-table td, .guidance-table td {
    padding: 0.5rem 1rem;
    border-bottom: 1px solid #2d3748;
    vertical-align: top;
}
.param-table td.param-name {
    font-family: "SFMono-Regular", Consolas, monospace;
    color: #63b3ed;
    white-space: nowrap;
    font-weight: 600;
}
.param-table td.param-value {
    font-family: "SFMono-Regular", Consolas, monospace;
    color: #68d391;
    white-space: nowrap;
}
.param-table td.param-constraint {
    font-family: "SFMono-Regular", Consolas, monospace;
    color: #f6ad55;
    white-space: nowrap;
}
.param-table td.param-description { color: #cbd5e0; }
.param-table tbody tr:hover,
.guidance-table tbody tr:hover { background: #1e2a3a; }
.guidance-table td.regime { color: #cbd5e0; white-space: nowrap; }
.guidance-table td.shrink-range { color: #f6ad55; font-weight: 600; }
.guidance-table td.notes { color: #a0aec0; }

/* ── Theory Section ───────────────────────────────── */
.theory-block {
    background: #1a202c;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    line-height: 1.8;
}
.theory-block h3 {
    color: #90cdf4;
    font-size: 1rem;
    font-weight: 600;
    margin: 1.25rem 0 0.5rem;
}
.theory-block h3:first-child { margin-top: 0; }
.theory-block p { color: #cbd5e0; margin-bottom: 0.75rem; }
.theory-block code {
    font-family: "SFMono-Regular", Consolas, monospace;
    background: #2d3748;
    padding: 0.1em 0.4em;
    border-radius: 4px;
    font-size: 0.875em;
    color: #63b3ed;
}
.theory-block .math-block {
    font-family: "SFMono-Regular", Consolas, monospace;
    background: #2d3748;
    border-left: 3px solid #4299e1;
    padding: 0.75rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.75rem 0;
    color: #e2e8f0;
    font-size: 0.9rem;
}
.theory-block ul {
    list-style: none;
    padding: 0;
}
.theory-block ul li {
    color: #cbd5e0;
    padding: 0.2rem 0;
    padding-left: 1.5rem;
    position: relative;
}
.theory-block ul li::before {
    content: "▸";
    position: absolute;
    left: 0;
    color: #4299e1;
}
.theory-block a { color: #63b3ed; }
.theory-block a:hover { text-decoration: underline; }
.refs { color: #a0aec0; font-size: 0.85rem; margin-top: 1rem; }

/* ── Footer ───────────────────────────────────────── */
.report-footer {
    text-align: center;
    padding: 1.5rem;
    color: #4a5568;
    font-size: 0.75rem;
    border-top: 1px solid #2d3748;
    margin-top: 3rem;
}
"""


# ── Parameter metadata ────────────────────────────────────────────────────────


def _constraint_str(field_info: object) -> str:
    """Extract a compact constraint string from a pydantic FieldInfo."""
    parts: list[str] = []
    # Pydantic v2 stores constraints inside field_info.metadata
    metadata = getattr(field_info, "metadata", [])
    for m in metadata:
        if hasattr(m, "gt") and m.gt is not None:
            parts.append(f"> {m.gt}")
        if hasattr(m, "ge") and m.ge is not None:
            parts.append(f"≥ {m.ge}")
        if hasattr(m, "lt") and m.lt is not None:
            parts.append(f"< {m.lt}")
        if hasattr(m, "le") and m.le is not None:
            parts.append(f"≤ {m.le}")
    return ", ".join(parts) if parts else "—"


def _fmt_value(v: object) -> str:
    """Format a config field value for display."""
    if isinstance(v, float):
        if v == int(v) and abs(v) >= 1e4:
            return f"{v:.2e}"
        if abs(v) < 0.01 and v != 0.0:
            return f"{v:.2e}"
        return f"{v:g}"
    return str(v)


def _params_table_html(config: BasanosConfig) -> str:
    """Render a styled HTML table of all BasanosConfig parameters.

    Args:
        config: The configuration instance to render.

    Returns:
        An HTML ``<table>`` string ready to embed in a page.
    """
    from .optimizer import BasanosConfig  # local import to avoid circularity

    rows: list[str] = []
    for name, field_info in BasanosConfig.model_fields.items():
        value = getattr(config, name)
        constraint = _constraint_str(field_info)
        description = field_info.description or "—"
        required = field_info.is_required()
        default_label = "required" if required else f"default: {_fmt_value(field_info.default)}"
        rows.append(
            f"<tr>"
            f'<td class="param-name">{name}</td>'
            f'<td class="param-value">{_fmt_value(value)}</td>'
            f'<td class="param-constraint">{constraint}</td>'
            f'<td class="param-description">{description}</td>'
            f'<td class="param-description" style="color:#718096;white-space:nowrap">{default_label}</td>'
            f"</tr>"
        )

    return (
        '<table class="param-table">'
        "<thead><tr>"
        "<th>Parameter</th>"
        "<th>Current&nbsp;Value</th>"
        "<th>Constraint</th>"
        "<th>Description</th>"
        "<th>Default</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


# ── Lambda-sweep chart ────────────────────────────────────────────────────────


def _lambda_sweep_fig(engine: BasanosEngine, n_points: int = 21) -> go.Figure:
    """Build a Plotly figure showing annualised Sharpe vs shrinkage weight λ.

    Args:
        engine: The engine to sweep.  All parameters other than ``shrink``
            are held fixed at their current values.
        n_points: Number of evenly-spaced λ values to evaluate in [0, 1].

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    lambdas = np.linspace(0.0, 1.0, n_points)
    sharpes = [engine.sharpe_at_shrink(float(lam)) for lam in lambdas]

    # Current config lambda marker
    current_lam = engine.cfg.shrink
    current_sharpe = engine.sharpe_at_shrink(current_lam)

    fig = go.Figure()

    # Main sweep line
    fig.add_trace(
        go.Scatter(
            x=list(lambdas),
            y=sharpes,
            mode="lines+markers",
            name="Sharpe(λ)",
            line={"color": "#4299e1", "width": 2},
            marker={"size": 5, "color": "#4299e1"},
            hovertemplate="λ = %{x:.2f}<br>Sharpe = %{y:.3f}<extra></extra>",
        )
    )

    # Current lambda marker
    fig.add_trace(
        go.Scatter(
            x=[current_lam],
            y=[current_sharpe],
            mode="markers",
            name=f"Current λ = {current_lam:.2f}",
            marker={"size": 12, "color": "#f6ad55", "symbol": "diamond"},
            hovertemplate=f"Current λ = {current_lam:.2f}<br>Sharpe = {current_sharpe:.3f}<extra></extra>",
        )
    )

    # Vertical reference lines at λ=0 and λ=1
    for x_val, label in [(0.0, "λ=0 (identity)"), (1.0, "λ=1 (no shrinkage)")]:
        fig.add_vline(
            x=x_val,
            line_dash="dash",
            line_color="#718096",
            annotation_text=label,
            annotation_position="top",
            annotation_font_color="#718096",
            annotation_font_size=10,
        )

    fig.update_layout(
        title={
            "text": "Annualised Sharpe Ratio vs Shrinkage Weight λ",
            "font": {"color": "#e2e8f0", "size": 15},
        },
        xaxis={
            "title": "Shrinkage weight λ  (0 = full identity, 1 = raw EWMA)",
            "color": "#a0aec0",
            "gridcolor": "#2d3748",
            "title_font": {"color": "#a0aec0"},
        },
        yaxis={
            "title": "Annualised Sharpe Ratio",
            "color": "#a0aec0",
            "gridcolor": "#2d3748",
            "title_font": {"color": "#a0aec0"},
        },
        paper_bgcolor="#1a202c",
        plot_bgcolor="#1a202c",
        font={"color": "#e2e8f0"},
        legend={"bgcolor": "#1a202c", "bordercolor": "#2d3748", "borderwidth": 1},
        margin={"t": 60, "b": 50, "l": 60, "r": 20},
    )
    return fig


# ── Guidance table ────────────────────────────────────────────────────────────

_GUIDANCE_ROWS = [
    ("n > 20, T < 40", "0.3 - 0.5", "Near-singular matrix likely; strong regularisation needed."),
    ("n ~= 10, T ~= 60", "0.5 - 0.7", "Balanced regime; moderate regularisation."),
    ("n < 10, T > 100", "0.7 - 0.9", "Well-conditioned sample; light shrinkage for stability."),
]


def _guidance_table_html() -> str:
    """Return an HTML table of shrinkage regime guidance (n / T heuristics)."""
    rows = "".join(
        f"<tr>"
        f'<td class="regime">{regime}</td>'
        f'<td class="shrink-range">{shrink_range}</td>'
        f'<td class="notes">{notes}</td>'
        f"</tr>"
        for regime, shrink_range, notes in _GUIDANCE_ROWS
    )
    return (
        '<table class="guidance-table">'
        "<thead><tr>"
        "<th>n (assets) / T (corr lookback)</th>"
        "<th>Suggested shrink (λ)</th>"
        "<th>Notes</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


# ── Theory HTML ───────────────────────────────────────────────────────────────

_THEORY_HTML = """
<div class="theory-block">
  <h3>Linear Shrinkage toward the Identity</h3>
  <p>
    The <code>shrink</code> parameter (&lambda;) controls how much the EWMA sample
    correlation matrix <em>C<sub>EWMA</sub></em> is regularised before being
    passed to the linear solver.  The shrunk matrix is:
  </p>
  <div class="math-block">C_shrunk = &lambda; &middot; C_EWMA + (1 - &lambda;) &middot; I_n</div>
  <p>
    where <em>I<sub>n</sub></em> is the n x n identity matrix.
    Setting &lambda; = 1 uses the raw EWMA correlation matrix (no shrinkage); setting
    &lambda; = 0 replaces it entirely with the identity (positions become purely
    signal-proportional, uncorrelated).
  </p>

  <h3>Why Shrinkage?</h3>
  <p>
    When the number of assets <em>n</em> is large relative to the lookback
    window <em>T</em> (high concentration ratio <em>n/T</em>), the sample
    covariance matrix is poorly estimated.  Extreme eigenvalues amplify
    estimation noise and cause the linear solver to allocate excessive
    leverage to a few eigendirections.  Shrinkage toward the identity damps
    these extremes, improves the condition number, and produces more stable,
    diversified positions.
  </p>

  <h3>When to Prefer Strong Shrinkage (low &lambda;)</h3>
  <ul>
    <li>Fewer than ~30 assets with a <code>corr</code> lookback shorter than 100 days.</li>
    <li>High-volatility or crisis regimes where correlations spike and the
        sample matrix is less representative of the true structure.</li>
    <li>Portfolios where estimation noise is more costly than correlation bias
        (low signal-to-noise ratio of <code>mu</code>).</li>
  </ul>

  <h3>When to Prefer Light Shrinkage (high &lambda;)</h3>
  <ul>
    <li>Many assets with a long lookback (low concentration ratio).</li>
    <li>The EWMA correlation structure carries genuine diversification
        information that the solver should exploit.</li>
    <li>Out-of-sample testing shows that position stability is not a concern.</li>
  </ul>

  <h3>EWMA Parameters - vola and corr</h3>
  <p>
    Both <code>vola</code> and <code>corr</code> are span-equivalent EWMA
    lookbacks (in trading periods).  The EWMA decay factor is
    <em>a = 2 / (span + 1)</em>, giving a centre-of-mass of
    <em>span / 2</em> periods.  <code>corr</code> must be &gt;= <code>vola</code>
    to ensure the correlation estimator sees at least as much history as the
    volatility normaliser.
  </p>

  <h3>References</h3>
  <p class="refs">
    Ledoit, O. &amp; Wolf, M. (2004).
    <em>A well-conditioned estimator for large-dimensional covariance matrices.</em>
    Journal of Multivariate Analysis, 88(2), 365-411.<br/>
    Chen, Y., Wiesel, A., Eldar, Y. C., &amp; Hero, A. O. (2010).
    <em>Shrinkage Algorithms for MMSE Covariance Estimation.</em>
    IEEE Transactions on Signal Processing, 58(10), 5016-5029.<br/>
    See also: <code>basanos.math._signal.shrink2id</code> for the implementation.
  </p>
</div>
"""


# ── Plotly helper ─────────────────────────────────────────────────────────────


def _figure_div(fig: go.Figure, include_plotlyjs: bool | str) -> str:
    """Return an HTML div string for *fig*."""
    return pio.to_html(fig, full_html=False, include_plotlyjs=include_plotlyjs)


# ── ConfigReport dataclass ────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ConfigReport:
    """Facade for generating HTML reports from a :class:`~basanos.math.optimizer.BasanosConfig`.

    Produces a self-contained, dark-themed HTML document with:

    * A **parameter table** listing all config fields, their current values,
      constraints, and descriptions.
    * An interactive **lambda-sweep chart** (requires *engine*) showing
      annualised Sharpe as a function of the shrinkage weight λ across [0, 1].
    * A **shrinkage-guidance table** mapping concentration-ratio regimes to
      suggested λ ranges.
    * A **theory section** covering Ledoit-Wolf linear shrinkage, EWMA
      parameter semantics, and academic references.

    Usage::

        # Static report (no lambda sweep) — from config alone
        report = config.report
        html_str = report.to_html()
        report.save("output/config_report.html")

        # Full report including lambda sweep — from engine
        report = engine.config_report
        report.save("output/config_report_with_sweep.html")
    """

    config: BasanosConfig
    engine: BasanosEngine | None = None

    def to_html(self, title: str = "Basanos Configuration Report") -> str:
        """Render a full HTML report as a string.

        The document is self-contained: Plotly.js is loaded from the CDN
        only when a lambda-sweep chart is included.  All other sections are
        pure HTML/CSS.

        Args:
            title: HTML ``<title>`` text and visible page heading.

        Returns:
            A complete HTML document as a :class:`str`.
        """
        cfg = self.config

        # ── Parameter table ────────────────────────────────────────────────
        params_html = _params_table_html(cfg)

        # ── Lambda sweep ───────────────────────────────────────────────────
        has_engine = self.engine is not None
        if has_engine:
            try:
                fig = _lambda_sweep_fig(self.engine)  # type: ignore[arg-type]
                sweep_div = _figure_div(fig, include_plotlyjs="cdn")
                sweep_section = f'<div class="chart-card">{sweep_div}</div>'
            except Exception as exc:
                sweep_section = f'<p class="chart-unavailable">Lambda sweep unavailable: {exc}</p>'
        else:
            sweep_section = (
                '<p class="chart-unavailable" style="padding:1.5rem;">'
                "Lambda sweep is available when accessing this report via "
                "<code>engine.config_report</code> (requires a "
                "<strong>BasanosEngine</strong> instance with price and signal data)."
                "</p>"
            )

        # ── Guidance table ─────────────────────────────────────────────────
        guidance_html = _guidance_table_html()

        # ── TOC links ──────────────────────────────────────────────────────
        toc_lambda = '<a href="#lambda-sweep">Lambda Sweep</a>' if has_engine else ""
        toc_extra_sep = "&nbsp;&nbsp;" if has_engine else ""

        # ── Assemble HTML ──────────────────────────────────────────────────
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
  <h1>&#x2699;&#xFE0F; {title}</h1>
  <div class="report-meta">
    <span><strong>vola:</strong> {cfg.vola}</span>
    <span><strong>corr:</strong> {cfg.corr}</span>
    <span><strong>clip:</strong> {cfg.clip}</span>
    <span><strong>shrink&nbsp;(λ):</strong> {cfg.shrink}</span>
    <span><strong>AUM:</strong> {cfg.aum:,.0f}</span>
  </div>
</header>

<nav class="toc">
  <a href="#parameters">Parameters</a>
  {toc_extra_sep}{toc_lambda}
  <a href="#guidance">Shrinkage Guidance</a>
  <a href="#theory">Theory</a>
</nav>

<div class="container">

  <section class="section" id="parameters">
    <h2 class="section-title">Configuration Parameters</h2>
    <div class="chart-card" style="overflow-x: auto;">{params_html}</div>
  </section>

  <section class="section" id="lambda-sweep">
    <h2 class="section-title">Lambda (Shrinkage) Sweep</h2>
    {sweep_section}
  </section>

  <section class="section" id="guidance">
    <h2 class="section-title">Shrinkage Guidance — n / T Regimes</h2>
    <div class="chart-card" style="overflow-x: auto;">{guidance_html}</div>
  </section>

  <section class="section" id="theory">
    <h2 class="section-title">Theory &amp; References</h2>
    {_THEORY_HTML}
  </section>

</div>

<footer class="report-footer">
  Generated by <strong>basanos</strong>
</footer>

</body>
</html>"""

    def save(self, path: str | Path, title: str = "Basanos Configuration Report") -> Path:
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
