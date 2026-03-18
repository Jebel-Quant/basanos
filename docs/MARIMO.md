# Marimo Notebooks

Interactive [Marimo](https://marimo.io/) notebooks for Basanos live under
`book/marimo/notebooks/`. This document is the index — what each notebook
does, how to run it, and what you need beforehand.

## Notebook Catalog

| Notebook | Description | Key concepts |
|---|---|---|
| [`demo.py`](#-demopy) | End-to-end interactive demo of the Basanos optimizer | Signal generation, correlation-aware position sizing, portfolio analytics, reactive UI |
| [`ewm_benchmark.py`](#-ewm_benchmarkpy) | Validates and benchmarks the NumPy/SciPy EWM correlation implementation against the legacy pandas version | EWM, `scipy.signal.lfilter`, NaN handling, performance comparison |
| [`shrinkage_guide.py`](#-shrinkage_guidepy) | Theoretical and empirical guide to tuning the shrinkage parameter λ | Marchenko-Pastur law, linear shrinkage `C(λ) = λ·C_EWMA + (1-λ)·I`, Sharpe vs. λ sweep, turnover analysis |

---

### 📊 `demo.py`

A comprehensive, fully reactive demo of the Basanos optimization pipeline.

**What it covers**

- Synthetic price data for 4 equity-like assets (GBM simulation)
- Momentum signal construction: `tanh(50 · (MA5 − MA20) / price)`
- Interactive sliders for `BasanosConfig` (`vola`, `corr`, `clip`, `shrink`)
- Volatility normalization → EWMA correlation → shrinkage → position solving (`C · x = μ`)
- AUM-scaled cash positions per asset
- Portfolio statistics: Sharpe, VaR, CVaR, drawdown
- Performance dashboard (NAV chart, lead/lag IR, correlation heatmap, tilt/timing decomposition)

**Prerequisites**: `basanos`, `numpy>=2.0`, `polars>=1.0`, `plotly>=6.0`

---

### ⚡ `ewm_benchmark.py`

Validates the NumPy/SciPy replacement for pandas-based EWM correlation.

**What it covers**

- Side-by-side correctness check: pandas `ewm().corr()` vs `scipy.signal.lfilter` IIR filter
- Tolerance gate: max absolute difference < 1×10⁻¹⁰
- NaN pattern comparison and heatmap visualisation
- Wall-clock performance benchmark across 5 realistic data sizes
- Interactive sliders for N (assets), T (timesteps), and `com` (centre-of-mass)

**Prerequisites**: `basanos`, `numpy>=2.0`, `polars>=1.0`, `pandas>=2.0`, `pyarrow>=12.0`, `plotly>=6.0`, `scipy>=1.0`

---

### 🔬 `shrinkage_guide.py`

A self-contained tutorial on choosing the shrinkage intensity λ.

**What it covers**

- Curse of dimensionality and the Marchenko-Pastur law
- Linear shrinkage formula: `C(λ) = λ·C_EWMA + (1−λ)·I_n`
- Short-lookback (noisy) vs. long-lookback (reliable) correlation regimes
- Sharpe ratio sweep over λ ∈ [0, 1]
- Mean daily turnover (% AUM) vs. λ
- Condition number analysis
- Practical rule-of-thumb: start at λ ≈ 1 − n/(2T), then tune on held-out data

**Prerequisites**: `basanos`, `numpy>=2.0`, `polars>=1.0`, `plotly>=6.0`

---

## Running the Notebooks

All notebooks are **pure Python files** with inline dependency metadata ([PEP 723](https://peps.python.org/pep-0723/)), so no manual environment setup is required.

### All notebooks at once (Makefile)

```bash
make marimo
```

This starts a headless Marimo editor server. Open `http://localhost:2718` in your browser to browse all notebooks.

### One notebook — interactive editor

```bash
marimo edit book/marimo/notebooks/demo.py
marimo edit book/marimo/notebooks/ewm_benchmark.py
marimo edit book/marimo/notebooks/shrinkage_guide.py
```

### One notebook — self-contained via uv (no prior install needed)

```bash
uv run book/marimo/notebooks/demo.py
uv run book/marimo/notebooks/ewm_benchmark.py
uv run book/marimo/notebooks/shrinkage_guide.py
```

`uv` reads the inline `# /// script` block, resolves and installs dependencies automatically, then launches the notebook.

### Read-only / presentation mode

```bash
marimo run book/marimo/notebooks/demo.py
```

### Validate all notebooks (CI-style)

```bash
make marimo-validate
```

Runs every notebook in a fresh isolated environment and stores outputs under `results/`.

---

## Environment Prerequisites

The notebooks are self-contained and install their own dependencies via `uv run`.
If you need the local `basanos` package itself (e.g. when editing source code and
re-running the notebook), install the project first:

```bash
make install
```

Python ≥ 3.11 and `uv` are the only system-level requirements.

---

## Notebook Structure

Marimo notebooks are **pure Python files** (`.py`), not JSON. Each file starts
with an inline dependency block:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.20.4",
#     "basanos",
#     "numpy>=2.0.0",
# ]
# [tool.uv.sources]
# basanos = { path = "../../..", editable = true }
# ///
```

This means:

- ✅ Easy version control with Git
- ✅ Standard code review workflows
- ✅ No hidden metadata
- ✅ Compatible with all Python tools

---

## Configuration

Marimo is configured in `pyproject.toml` to import the local package:

```toml
[tool.marimo.runtime]
pythonpath = ["src"]
```

---

## CI/CD Integration

The `.github/workflows/marimo.yml` workflow automatically:

1. Discovers all `.py` files in `book/marimo/notebooks/`
2. Runs each notebook in a fresh environment
3. Verifies that notebooks can bootstrap themselves
4. Ensures reproducibility

---

## Creating New Notebooks

1. Create a new `.py` file in `book/marimo/notebooks/`:
   ```bash
   marimo edit book/marimo/notebooks/my_notebook.py
   ```

2. Add inline metadata at the top:
   ```python
   # /// script
   # requires-python = ">=3.11"
   # dependencies = [
   #     "marimo==0.20.4",
   #     # ... other dependencies
   # ]
   # ///
   ```

3. Build your notebook with cells.

4. Test it runs in a clean environment:
   ```bash
   uv run book/marimo/notebooks/my_notebook.py
   ```

5. Add an entry to the [Notebook Catalog](#notebook-catalog) table above.

6. Commit and push — CI will validate it automatically.

---

## Learn More

- **Marimo Documentation**: [https://docs.marimo.io/](https://docs.marimo.io/)
- **Example Gallery**: [https://marimo.io/examples](https://marimo.io/examples)
- **Community Discord**: [https://discord.gg/JE7nhX6mD8](https://discord.gg/JE7nhX6mD8)

## Tips

- **Reactivity**: Cells automatically re-run when their dependencies change
- **Pure Python**: Edit notebooks in any text editor, not just Marimo's UI
- **Git-Friendly**: Notebooks diff and merge like regular Python files
- **Self-Contained**: Inline metadata makes notebooks reproducible without a shared venv
- **Interactive**: Use Marimo's rich UI components (sliders, dropdowns, tabs) for better UX

---

*Happy exploring with Marimo! 🚀*
