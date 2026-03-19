# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.20.4",
#     "basanos",
#     "numpy>=2.0.0",
#     "plotly>=6.0.0",
# ]
# [tool.uv.sources]
# basanos = { path = "../../..", editable = true }
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from basanos.math import FactorModel


@app.cell
def cell_01():
    """Render the factor model guide introduction."""
    mo.md(
        r"""
        # 🧮 Basanos — Factor Risk Model Guide

        The **`FactorModel`** class encapsulates a *factor risk model* decomposition
        that expresses an asset covariance matrix as a low-rank systematic component
        plus a diagonal idiosyncratic term:

        $$\bm{\Sigma} = \mathbf{B}\mathbf{F}\mathbf{B}^\top + \mathbf{D}$$

        where

        - $\mathbf{B} \in \mathbb{R}^{n \times k}$ — **factor loading matrix**: each column
          gives the sensitivity of all $n$ assets to one latent factor.
        - $\mathbf{F} \in \mathbb{R}^{k \times k}$ — **factor covariance matrix** (positive
          definite): how the $k$ factors co-vary.
        - $\mathbf{D} = \operatorname{diag}(d_1, \dots, d_n)$, $d_i > 0$ — **idiosyncratic
          variance**: asset-specific risk *unexplained* by the common factors.

        The key assumption is $k \ll n$: dominant risk sources are captured by a handful
        of factors, making the model both interpretable and computationally efficient.

        ## What this notebook covers

        1. 🏗️ **Direct construction** — build a `FactorModel` from explicit arrays
        2. 📐 **Properties** — `n_assets`, `n_factors`, and reconstructed `covariance`
        3. 📈 **Fitting from returns** — `FactorModel.from_returns()` via truncated SVD
        4. 🎚️ **Interactive k selection** — visualise the singular value spectrum to pick $k$
        5. 🎨 **Factor structure** — heatmap of factor loadings and idiosyncratic variances
        6. 🔍 **Covariance approximation** — how well the rank-*k* model captures correlations
        7. ⚡ **Woodbury solve** — efficient $\bm{\Sigma}^{-1}\mathbf{b}$ without forming $\bm{\Sigma}$
        """
    )


@app.cell
def cell_02():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_03():
    """Render the direct construction section."""
    mo.md(
        r"""
        ## 🏗️ Direct Construction

        A `FactorModel` is a **frozen dataclass** — once created its fields cannot be
        mutated.  Pass the three arrays explicitly to the constructor.  The
        `__post_init__` validator checks shape consistency and strict positivity of
        the idiosyncratic variances.
        """
    )


@app.cell
def cell_04():
    """Demonstrate direct FactorModel construction."""
    # 3 assets, 2 factors — minimal illustrative example
    _loadings = np.array(
        [
            [0.8, 0.2],  # asset 1: mostly factor 1
            [0.3, 0.7],  # asset 2: mixed
            [0.1, 0.9],  # asset 3: mostly factor 2
        ]
    )
    _factor_cov = np.array([[0.04, 0.01], [0.01, 0.03]])  # 2x2 factor covariance
    _idio_var = np.array([0.02, 0.015, 0.025])  # idiosyncratic variances

    fm_manual = FactorModel(
        factor_loadings=_loadings,
        factor_covariance=_factor_cov,
        idiosyncratic_var=_idio_var,
    )

    mo.callout(
        mo.md(
            f"""
            **Constructed `FactorModel`**

            | Property | Value |
            |---|---|
            | `n_assets` | `{fm_manual.n_assets}` |
            | `n_factors` | `{fm_manual.n_factors}` |
            | `factor_loadings.shape` | `{fm_manual.factor_loadings.shape}` |
            | `factor_covariance.shape` | `{fm_manual.factor_covariance.shape}` |
            | `idiosyncratic_var.shape` | `{fm_manual.idiosyncratic_var.shape}` |
            | `covariance.shape` | `{fm_manual.covariance.shape}` |

            The frozen dataclass guarantees that none of these fields can be
            overwritten after construction — immutability is enforced by Python.
            """
        ),
        kind="success",
    )
    return (fm_manual,)


@app.cell
def cell_05(fm_manual):
    """Show the reconstructed full covariance matrix."""
    _cov = fm_manual.covariance
    _fig = go.Figure(
        go.Heatmap(
            z=_cov,
            x=["Asset 1", "Asset 2", "Asset 3"],
            y=["Asset 1", "Asset 2", "Asset 3"],
            colorscale="RdBu_r",
            zmid=0,
            text=[[f"{v:.4f}" for v in row] for row in _cov],
            texttemplate="%{text}",
        )
    )
    _fig.update_layout(
        title="Reconstructed covariance matrix  Σ = BFBᵀ + D  (manual example)",
        height=320,
    )
    mo.vstack(
        [
            mo.md("### Reconstructed Covariance Matrix"),
            mo.ui.plotly(_fig),
            mo.md(
                r"*Off-diagonal entries reflect systematic co-movement through shared factor loadings.  "
                r"Diagonal = systematic variance + idiosyncratic variance.*"
            ),
        ]
    )


@app.cell
def cell_06():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_07():
    """Introduce the from_returns section."""
    mo.md(
        r"""
        ## 📈 Fitting from Returns: `FactorModel.from_returns()`

        In practice the factor model is *estimated* from a return matrix
        $\mathbf{R} \in \mathbb{R}^{T \times n}$ using the **Singular Value Decomposition**:

        $$\mathbf{R} = \mathbf{U}\bm{\Sigma}\mathbf{V}^\top$$

        The top-$k$ right singular vectors define the factor loadings,
        and the top-$k$ singular values define the factor covariance:

        $$\mathbf{B} = \mathbf{V}_k, \quad \mathbf{F} = \bm{\Sigma}_k^2 / T, \quad
          \hat{d}_i = \max\!\bigl(\hat{\sigma}_i^2 - (\mathbf{B}\mathbf{F}\mathbf{B}^\top)_{ii},\, \varepsilon\bigr)$$

        When the return columns have **unit variance** (i.e. $\hat{\sigma}_i^2 = 1$,
        as produced by volatility adjustment), this simplifies to
        $\hat{d}_i = \max(1 - (\mathbf{B}\mathbf{F}\mathbf{B}^\top)_{ii}, \varepsilon)$
        and the reconstructed covariance also has unit diagonal.  The example below
        uses standardised returns to satisfy this assumption.

        Below we generate **synthetic correlated returns** for 8 assets over 200 days,
        then fit a `FactorModel` and inspect the result.
        """
    )


@app.cell
def cell_08():
    """Generate synthetic correlated returns for 8 assets."""
    _rng = np.random.default_rng(42)
    _n_assets = 8
    _t_len = 200

    # Build a low-rank true covariance: 3 underlying factors drive the assets
    _true_loadings = _rng.standard_normal((_n_assets, 3))
    _true_cov = _true_loadings @ _true_loadings.T + np.diag(np.ones(_n_assets) * 0.5)

    # Cholesky-sample correlated returns, then standardise to unit variance
    _chol = np.linalg.cholesky(_true_cov)
    _raw = _rng.standard_normal((_t_len, _n_assets)) @ _chol.T
    returns = _raw / _raw.std(axis=0, keepdims=True)  # unit-variance columns

    _asset_names = [f"A{i + 1}" for i in range(_n_assets)]
    mo.callout(
        mo.md(
            f"""
            **Synthetic return matrix**

            - Shape: `{returns.shape}` — {_t_len} days × {_n_assets} assets
            - Column means ≈ 0, column stds ≈ 1 (standardised)
            - True underlying structure: **3 latent factors**
            """
        ),
        kind="info",
    )
    return _asset_names, returns


@app.cell
def cell_09():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_10():
    """Introduce the singular value spectrum section."""
    mo.md(
        r"""
        ## 🎚️ Choosing *k*: The Singular Value Spectrum

        A useful diagnostic for selecting the number of factors $k$ is the
        **singular value spectrum** of the return matrix.  Genuine risk factors
        show up as large singular values well separated from the bulk of the
        spectrum (noise floor).

        Use the slider below to choose $k$ and observe how the cumulative
        explained variance changes.
        """
    )


@app.cell
def cell_11(returns):
    """Compute singular values and show the spectrum."""
    _, sv, _ = np.linalg.svd(returns, full_matrices=False)
    _explained = (sv**2) / (sv**2).sum()
    _cumulative = np.cumsum(_explained)

    _fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Singular Value Spectrum", "Cumulative Explained Variance"],
    )
    _fig.add_trace(
        go.Bar(
            x=list(range(1, len(sv) + 1)),
            y=sv.tolist(),
            name="Singular value",
            marker_color="#2980b9",
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=list(range(1, len(_cumulative) + 1)),
            y=(_cumulative * 100).tolist(),
            mode="lines+markers",
            name="Cumulative variance (%)",
            line={"color": "#e74c3c", "width": 2},
        ),
        row=1,
        col=2,
    )
    _fig.add_hline(y=80, line_dash="dash", line_color="grey", annotation_text="80 %", row=1, col=2)
    _fig.update_layout(
        height=380,
        showlegend=False,
        xaxis_title="Factor index",
        xaxis2_title="Number of factors k",
        yaxis_title="Singular value",
        yaxis2_title="Cumulative explained variance (%)",
    )
    mo.ui.plotly(_fig)
    return (sv,)


@app.cell
def cell_12():
    """Create slider for k selection."""
    k_slider = mo.ui.slider(
        start=1,
        stop=8,
        value=3,
        step=1,
        label="Number of factors k:",
        show_value=True,
    )
    mo.vstack([k_slider])
    return (k_slider,)


@app.cell
def cell_13(k_slider, returns, sv):
    """Fit FactorModel with chosen k and display summary."""
    fm = FactorModel.from_returns(returns, k=k_slider.value)

    _explained_k = ((sv[: k_slider.value] ** 2).sum() / (sv**2).sum()) * 100

    mo.callout(
        mo.md(
            f"""
            **Fitted `FactorModel` with k = {k_slider.value}**

            | Property | Value |
            |---|---|
            | `n_assets` | `{fm.n_assets}` |
            | `n_factors` | `{fm.n_factors}` |
            | `factor_loadings.shape` | `{fm.factor_loadings.shape}` |
            | `factor_covariance.shape` | `{fm.factor_covariance.shape}` |
            | Explained variance | **{_explained_k:.1f} %** |
            | Mean idiosyncratic var | `{fm.idiosyncratic_var.mean():.4f}` |
            """
        ),
        kind="success",
    )
    return (fm,)


@app.cell
def cell_14():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_15():
    """Introduce the factor structure section."""
    mo.md(
        r"""
        ## 🎨 Factor Structure

        The two charts below show the **factor loading matrix** $\mathbf{B}$ and the
        **idiosyncratic variance** $\mathbf{d}$ for the currently selected $k$.

        - **Factor loadings heatmap** — each cell $(i, j)$ is the sensitivity of
          asset $i$ to factor $j$.  Large absolute values indicate the asset is
          strongly driven by that factor.
        - **Idiosyncratic variance** — the residual per-asset variance not captured
          by the $k$ common factors.  A near-zero bar means the asset is almost
          entirely explained by the factor model.
        """
    )


@app.cell
def cell_16(fm, k_slider):
    """Plot factor loadings heatmap and idiosyncratic variance bar chart."""
    _asset_labels = [f"A{i + 1}" for i in range(fm.n_assets)]
    _factor_labels = [f"F{j + 1}" for j in range(fm.n_factors)]

    _fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Factor Loadings B  (n={fm.n_assets}, k={k_slider.value})",
            "Idiosyncratic Variance d",
        ],
        column_widths=[0.6, 0.4],
    )

    # Factor loadings heatmap
    _fig.add_trace(
        go.Heatmap(
            z=fm.factor_loadings.tolist(),
            x=_factor_labels,
            y=_asset_labels,
            colorscale="RdBu_r",
            zmid=0,
            text=[[f"{v:.3f}" for v in row] for row in fm.factor_loadings],
            texttemplate="%{text}",
            showscale=True,
            name="Loadings",
        ),
        row=1,
        col=1,
    )

    # Idiosyncratic variance bar chart
    _fig.add_trace(
        go.Bar(
            x=_asset_labels,
            y=fm.idiosyncratic_var.tolist(),
            name="Idiosyncratic var",
            marker_color="#8e44ad",
            text=[f"{v:.4f}" for v in fm.idiosyncratic_var],
            textposition="outside",
        ),
        row=1,
        col=2,
    )

    _fig.update_layout(
        height=420,
        showlegend=False,
        yaxis2_title="Variance",
        xaxis2_title="Asset",
    )
    mo.ui.plotly(_fig)


@app.cell
def cell_17():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_18():
    """Introduce the covariance approximation section."""
    mo.md(
        r"""
        ## 🔍 Covariance Approximation Quality

        The factor model provides a **rank-$k$ approximation** of the full sample
        covariance matrix.  The heatmaps below compare:

        - **Sample correlation** $\hat{C}$ — empirical correlations from the return matrix.
        - **Factor-model correlation** $\hat{C}^{(k)}$ — correlations implied by
          $\mathbf{B}\mathbf{F}\mathbf{B}^\top + \mathbf{D}$ (normalised to unit diagonal).
        - **Approximation error** — absolute difference $|\hat{C} - \hat{C}^{(k)}|$.

        A small $k$ captures only the dominant systematic structure; the off-diagonal
        entries of the error matrix show which asset pairs are under-modelled.
        """
    )


@app.cell
def cell_19(fm, returns):
    """Plot sample vs factor-model correlation and approximation error."""
    _sample_cov = np.cov(returns.T)
    _sample_std = np.sqrt(np.diag(_sample_cov))
    _sample_corr = _sample_cov / np.outer(_sample_std, _sample_std)

    # Factor-model implied correlation (normalised to unit diagonal)
    _fm_cov = fm.covariance
    _fm_std = np.sqrt(np.diag(_fm_cov))
    _fm_corr = _fm_cov / np.outer(_fm_std, _fm_std)

    _err = np.abs(_sample_corr - _fm_corr)
    _asset_labels = [f"A{i + 1}" for i in range(fm.n_assets)]

    _fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Sample Correlation  Ĉ",
            f"Factor-Model Correlation  Ĉ⁽ᵏ⁾  (k={fm.n_factors})",
            "Absolute Error  |Ĉ − Ĉ⁽ᵏ⁾|",
        ],
    )

    for _col_idx, (_mat, _scale, _title) in enumerate(
        [
            (_sample_corr, "RdBu_r", "Sample"),
            (_fm_corr, "RdBu_r", "Factor"),
            (_err, "Oranges", "Error"),
        ],
        start=1,
    ):
        _fig.add_trace(
            go.Heatmap(
                z=_mat.tolist(),
                x=_asset_labels,
                y=_asset_labels,
                colorscale=_scale,
                zmid=0 if _scale == "RdBu_r" else None,
                zmin=-1 if _scale == "RdBu_r" else 0,
                zmax=1 if _scale == "RdBu_r" else None,
                showscale=True,
                name=_title,
            ),
            row=1,
            col=_col_idx,
        )

    _mae_offdiag = float(_err[np.triu_indices_from(_err, k=1)].mean())
    _fig.update_layout(
        height=380,
        showlegend=False,
        title=f"Mean absolute off-diagonal error: {_mae_offdiag:.4f}",
    )
    mo.ui.plotly(_fig)


@app.cell
def cell_20():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_21():
    """Introduce the Woodbury solve section."""
    mo.md(
        r"""
        ## ⚡ Efficient Solve via the Woodbury Identity

        The `FactorModel.solve(b)` method solves the linear system
        $\bm{\Sigma}\mathbf{x} = \mathbf{b}$ **without forming the full
        $n \times n$ matrix**, using the Sherman–Morrison–Woodbury formula:

        $$(\mathbf{D} + \mathbf{B}\mathbf{F}\mathbf{B}^\top)^{-1}
          = \mathbf{D}^{-1}
            - \mathbf{D}^{-1}\mathbf{B}
              \underbrace{(\mathbf{F}^{-1} + \mathbf{B}^\top\mathbf{D}^{-1}\mathbf{B})}_{k \times k}{}^{-1}
              \mathbf{B}^\top\mathbf{D}^{-1}$$

        | Step | Cost |
        |---|---|
        | $\mathbf{D}^{-1}$ (diagonal inversion) | $O(n)$ |
        | Inner $k \times k$ system | $O(k^3 + kn)$ |
        | Total | $O(k^3 + kn)$  vs.  $O(n^3)$ for direct inversion |

        For $k \ll n$ this is a substantial saving — particularly relevant when
        `solve` is called at every timestamp inside the optimizer loop.

        The cell below verifies that the Woodbury result matches direct inversion
        numerically.
        """
    )


@app.cell
def cell_22(fm, returns):
    """Verify Woodbury solve matches direct inversion."""
    _rng_solve = np.random.default_rng(99)
    _b_vec = _rng_solve.standard_normal(fm.n_assets)

    # Woodbury solve
    _x_woodbury = fm.solve(_b_vec)

    # Direct solve via full covariance
    _x_direct = np.linalg.solve(fm.covariance, _b_vec)

    _max_err = float(np.abs(_x_woodbury - _x_direct).max())
    _rel_err = float(np.abs(_x_woodbury - _x_direct).max() / (np.abs(_x_direct).max() + 1e-16))
    _residual = float(np.abs(fm.covariance @ _x_woodbury - _b_vec).max())

    mo.callout(
        mo.md(
            f"""
            **Woodbury solve vs direct inversion**

            | Check | Value |
            |---|---|
            | Max absolute error | `{_max_err:.2e}` |
            | Max relative error | `{_rel_err:.2e}` |
            | Residual  max ‖Σx − b‖ | `{_residual:.2e}` |

            ✅ Both solutions agree to machine precision.
            The Woodbury result satisfies $\\bm{{\\Sigma}}\\mathbf{{x}} = \\mathbf{{b}}$ exactly
            (up to floating-point rounding).
            """
        ),
        kind="success",
    )


@app.cell
def cell_23():
    """Render separator."""
    mo.md(r"""---""")


@app.cell
def cell_24():
    """Render the conclusion."""
    mo.md(
        r"""
        ## 🎉 Conclusion

        This notebook demonstrated the full lifecycle of the `FactorModel` class:

        ✅ **Direct construction** — provide $\mathbf{B}$, $\mathbf{F}$, and $\mathbf{d}$ explicitly;
        the frozen dataclass validates shape and positivity on construction.

        ✅ **Fitting from returns** — `FactorModel.from_returns(R, k)` extracts the top-$k$
        SVD components to produce a calibrated low-rank covariance model.

        ✅ **Singular value spectrum** — inspect explained variance to choose $k$ objectively;
        a sharp drop in singular values signals where the noise floor begins.

        ✅ **Factor structure** — the loading heatmap reveals which assets are dominated by which
        factors; idiosyncratic variances show what the model leaves unexplained.

        ✅ **Covariance approximation** — a rank-$k$ model recovers the dominant correlation
        structure with a controllable approximation error.

        ✅ **Woodbury solve** — `fm.solve(b)` solves $\bm{\Sigma}\mathbf{x} = \mathbf{b}$ in
        $O(k^3 + kn)$ time instead of $O(n^3)$, with numerically identical results.

        ### Next steps

        - Use `FactorModel` inside `BasanosConfig` via the `k` parameter to enable
          factor-model covariance estimation in the full optimizer pipeline.
        - Compare `covariance_mode="pca"` (factor-model based) with `"ewma_shrink"` using
          `BasanosEngine.sharpe_at_pca_components()`.
        - See the [basanos repository](https://github.com/Jebel-Quant/basanos) for the
          full API reference and the `demo` notebook for an end-to-end portfolio example.
        """
    )


if __name__ == "__main__":
    app.run()
