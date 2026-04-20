"""HTML report generation for BasanosConfig parameter analysis.

This module defines the `ConfigReport` facade which produces a
self-contained HTML document summarising all configuration parameters,
their constraints and descriptions, an interactive lambda-sweep chart
(when a `BasanosEngine` is provided), a
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
from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    from .optimizer import BasanosConfig, BasanosEngine

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    autoescape=select_autoescape(["html"]),
)


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
        A `Figure`.
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


# ── Plotly helper ─────────────────────────────────────────────────────────────


def _figure_div(fig: go.Figure, include_plotlyjs: bool | str) -> str:
    """Return an HTML div string for *fig*."""
    return pio.to_html(fig, full_html=False, include_plotlyjs=include_plotlyjs)


# ── ConfigReport dataclass ────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ConfigReport:
    """Facade for generating HTML reports from a `BasanosConfig`.

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
            A complete HTML document as a `str`.
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

        # ── Render template ────────────────────────────────────────────────
        template = _env.get_template("config_report.html")
        return template.render(
            title=title,
            vola=cfg.vola,
            corr=cfg.corr,
            clip=cfg.clip,
            shrink=cfg.shrink,
            aum=f"{cfg.aum:,.0f}",
            toc_lambda=toc_lambda,
            toc_extra_sep=toc_extra_sep,
            params_html=params_html,
            sweep_section=sweep_section,
            guidance_html=guidance_html,
            container_max_width="1200px",
        )

    def save(self, path: str | Path, title: str = "Basanos Configuration Report") -> Path:
        """Save the HTML report to a file.

        A ``.html`` suffix is appended automatically when *path* has no
        file extension.

        Args:
            path: Destination file path.
            title: HTML ``<title>`` text and visible page heading.

        Returns:
            The resolved `Path` of the written file.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".html")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_html(title=title), encoding="utf-8")
        return p
