"""Plotting utilities for portfolio analytics using Plotly.

This module defines the Plots facade which renders common portfolio visuals
such as snapshots, lagged performance curves, smoothed-holdings curves, and
lead/lag information ratio bar charts. Designed for notebook use.

Examples:
    >>> import dataclasses
    >>> from basanos.analytics._plots import Plots
    >>> dataclasses.is_dataclass(Plots)
    True
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    # Import the local Portfolio type for type checking and documentation tools.
    from .portfolio import Portfolio

# Ensure Plotly works with Marimo (set after imports to satisfy linters)
pio.renderers.default = "plotly_mimetype"


@dataclasses.dataclass(frozen=True)
class Plots:
    """Facade for portfolio plots built with Plotly.

    Provides convenience methods to visualize portfolio performance and
    diagnostics directly from a Portfolio instance (e.g., snapshot charts,
    lagged performance, smoothed holdings, and lead/lag IR).
    """

    portfolio: Portfolio

    def lead_lag_ir_plot(self, start: int = -10, end: int = 19) -> go.Figure:
        """Plot Sharpe ratio (IR) across lead/lag variants of the portfolio.

        Builds portfolios with cash positions lagged from ``start`` to ``end``
        (inclusive) and plots a bar chart of the Sharpe ratio for each lag.
        Positive lags delay weights; negative lags lead them.

        Args:
            start: First lag to include (default: -10).
            end: Last lag to include (default: +19).

        Returns:
            A Plotly Figure with one bar per lag labeled by the lag value.
        """
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError
        if start > end:
            start, end = end, start

        lags = list(range(start, end + 1))

        x_vals: list[int] = []
        y_vals: list[float] = []

        for n in lags:
            pf = self.portfolio if n == 0 else self.portfolio.lag(n)
            # Compute Sharpe on the portfolio's returns series
            sharpe_val = pf.stats.sharpe().get("returns", float("nan"))
            # Ensure a float (Stats returns mapping asset->value)
            y_vals.append(float(sharpe_val) if sharpe_val is not None else float("nan"))
            x_vals.append(n)

        fig = go.Figure(
            data=[
                go.Bar(x=x_vals, y=y_vals, name="Sharpe by lag", marker_color="#1f77b4"),
            ]
        )
        fig.update_layout(
            title="Lead/Lag Information Ratio (Sharpe) by Lag",
            xaxis_title="Lag (steps)",
            yaxis_title="Sharpe ratio",
            plot_bgcolor="white",
            hovermode="x",
        )
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
        return fig

    def snapshot(self, log_scale: bool = False) -> go.Figure:
        """Return a snapshot dashboard of NAV and drawdown.

        Args:
            log_scale (bool, optional): If True, display NAV on a log scale. Defaults to False.

        Returns:
            plotly.graph_objects.Figure: A Figure with accumulated NAV (including tilt/timing)
                and drawdown shaded area, equipped with a range selector.
        """
        # Create subplot grid with domain for stats table
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.66, 0.33],
            subplot_titles=["Accumulated Profit", "Drawdown"],
            vertical_spacing=0.05,
        )

        # --- Row 1: Cumulative Returns
        fig.add_trace(
            go.Scatter(
                x=self.portfolio.nav_accumulated["date"],
                y=self.portfolio.nav_accumulated["NAV_accumulated"],
                mode="lines",
                name="NAV",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.portfolio.tilt.nav_accumulated["date"],
                y=self.portfolio.tilt.nav_accumulated["NAV_accumulated"],
                mode="lines",
                name="Tilt",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.portfolio.timing.nav_accumulated["date"],
                y=self.portfolio.timing.nav_accumulated["NAV_accumulated"],
                mode="lines",
                name="Timing",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.portfolio.drawdown["date"],
                y=self.portfolio.drawdown["drawdown_pct"],
                mode="lines",
                fill="tozeroy",
                name="Drawdown",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_hline(y=0, line_width=1, line_color="gray", row=2, col=1)

        # Layout
        fig.update_layout(
            title="Performance Dashboard",
            height=1200,
            hovermode="x unified",
            plot_bgcolor="white",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            xaxis={
                "rangeselector": {
                    "buttons": [
                        {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                        {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                        {"count": 3, "label": "3y", "step": "year", "stepmode": "backward"},
                        {"step": "year", "stepmode": "todate", "label": "YTD"},
                        {"step": "all", "label": "All"},
                    ]
                },
                "rangeslider": {"visible": False},
                "type": "date",
            },
        )

        fig.update_yaxes(title_text="NAV (accumulated)", row=1, col=1, tickformat=".2s")
        fig.update_yaxes(title_text="Drawdown", row=2, col=1, tickformat=".0%")

        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")

        if log_scale:
            fig.update_yaxes(type="log", row=1, col=1)
            # Ensure the first y-axis is explicitly set for environments
            # where subplot updates may not propagate to layout alias.
            if hasattr(fig.layout, "yaxis"):
                fig.layout.yaxis.type = "log"

        return fig

    def lagged_performance_plot(self, lags: list[int] | None = None, log_scale: bool = False) -> go.Figure:
        """Plot NAV_accumulated for multiple lagged portfolios.

        Creates a Plotly figure with one line per lag value showing the
        accumulated NAV series for the portfolio with cash positions
        shifted by that lag. By default, lags [0, 1, 2, 3, 4] are used.

        Args:
            lags: A list of integer lags to apply; defaults to [0, 1, 2, 3, 4].
            log_scale: If True, set the primary y-axis to logarithmic scale.

        Returns:
            A Plotly Figure containing one trace per requested lag.
        """
        if lags is None:
            lags = [0, 1, 2, 3, 4]
        if not isinstance(lags, list) or not all(isinstance(x, int) for x in lags):
            raise TypeError

        fig = go.Figure()
        for lag in lags:
            pf = self.portfolio if lag == 0 else self.portfolio.lag(lag)
            nav = pf.nav_accumulated
            fig.add_trace(
                go.Scatter(
                    x=nav["date"],
                    y=nav["NAV_accumulated"],
                    mode="lines",
                    name=f"lag {lag}",
                    line={"width": 1},
                )
            )

        fig.update_layout(
            title="NAV accumulated by lag",
            hovermode="x unified",
            plot_bgcolor="white",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            xaxis={
                "rangeselector": {
                    "buttons": [
                        {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                        {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                        {"count": 3, "label": "3y", "step": "year", "stepmode": "backward"},
                        {"step": "year", "stepmode": "todate", "label": "YTD"},
                        {"step": "all", "label": "All"},
                    ]
                },
                "rangeslider": {"visible": False},
                "type": "date",
            },
        )
        fig.update_yaxes(title_text="NAV (accumulated)")
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")

        if log_scale:
            fig.update_yaxes(type="log")
            if hasattr(fig.layout, "yaxis"):
                fig.layout.yaxis.type = "log"

        return fig

    def correlation_heatmap(
        self,
        frame: pl.DataFrame | None = None,
        name: str = "portfolio",
        title: str = "Correlation heatmap",
    ) -> go.Figure:
        """Plot a correlation heatmap for assets and the portfolio.

        If ``frame`` is None, uses the portfolio's prices. The portfolio's
        profit series is appended under ``name`` before computing the
        correlation matrix.

        Args:
            frame: Optional Polars DataFrame with at least the asset price
                columns. If omitted, uses ``self.portfolio.prices``.
            name: Column name under which to include the portfolio profit.
            title: Plot title.

        Returns:
            A Plotly Figure rendering the correlation matrix as a heatmap.
        """
        if frame is None:
            frame = self.portfolio.prices

        corr = self.portfolio.correlation(frame, name=name)

        # Create an interactive heatmap
        fig = px.imshow(
            corr,
            x=corr.columns,
            y=corr.columns,
            text_auto=".2f",  # show correlation values
            color_continuous_scale="RdBu_r",  # red-blue diverging colormap
            zmin=-1,
            zmax=1,  # correlation range
            title=title,
        )

        # Adjust layout
        fig.update_layout(
            xaxis_title="", yaxis_title="", width=700, height=600, coloraxis_colorbar={"title": "Correlation"}
        )

        return fig

    def smoothed_holdings_performance_plot(
        self,
        windows: list[int] | None = None,
        log_scale: bool = False,
    ) -> go.Figure:
        """Plot NAV_accumulated for smoothed-holding portfolios.

        Builds portfolios with cash positions smoothed by a trailing rolling
        mean over the previous ``n`` steps (window size n+1) for n in
        ``windows`` (defaults to [0, 1, 2, 3, 4]) and plots their
        accumulated NAV curves.

        Args:
            windows: List of non-negative integers specifying smoothing steps
                to include; defaults to [0, 1, 2, 3, 4].
            log_scale: If True, set the primary y-axis to logarithmic scale.

        Returns:
            A Plotly Figure containing one line per requested smoothing level.
        """
        if windows is None:
            windows = [0, 1, 2, 3, 4]
        if not isinstance(windows, list) or not all(isinstance(x, int) and x >= 0 for x in windows):
            raise TypeError

        fig = go.Figure()
        for n in windows:
            pf = self.portfolio if n == 0 else self.portfolio.smoothed_holding(n)
            nav = pf.nav_accumulated
            fig.add_trace(
                go.Scatter(
                    x=nav["date"],
                    y=nav["NAV_accumulated"],
                    mode="lines",
                    name=f"smooth {n}",
                    line={"width": 1},
                )
            )

        fig.update_layout(
            title="NAV accumulated by smoothed holdings",
            hovermode="x unified",
            plot_bgcolor="white",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            xaxis={
                "rangeselector": {
                    "buttons": [
                        {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                        {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                        {"count": 3, "label": "3y", "step": "year", "stepmode": "backward"},
                        {"step": "year", "stepmode": "todate", "label": "YTD"},
                        {"step": "all", "label": "All"},
                    ]
                },
                "rangeslider": {"visible": False},
                "type": "date",
            },
        )
        fig.update_yaxes(title_text="NAV (accumulated)")
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgrey")

        if log_scale:
            fig.update_yaxes(type="log")
            if hasattr(fig.layout, "yaxis"):
                fig.layout.yaxis.type = "log"

        return fig
