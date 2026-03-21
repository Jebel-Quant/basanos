# Road to 10/10: basanos Improvement Plan

_Last updated: 2026-03-21 — reflects 5th quality analysis_

---

## Current Scores

| Category | Score | Status |
|---|---|---|
| Code Quality | 8/10 | ~40 lines of duplication remain in `_iter_solve`; no `TypeAlias` annotations |
| Documentation | 9/10 | No hosted API reference site |
| Testing | 8/10 | No mutation testing; no numerical stability test suite |
| Architecture & Design | 9/10 | Generator yield tuples lack `TypeAlias` annotations |
| CI/CD & DevOps | **10/10** | COMPLETE |
| Mathematical Rigor | 8/10 | `min_corr_denom` unjustified; no large-universe warning/guidance |
| Dependencies & Packaging | 9/10 | Not on conda-forge |
| Security | 9/10 | Semgrep (`p/numpy`) absent |
| Community & Maintenance | 7/10 | No PR template; no repo topics; no external adoption yet |
| Performance | 8/10 | No chunked path for N > 150; no optional Numba acceleration |

---

## Completed Items

The following items from the original plan are done and require no further action:

- **1a** `SolveStatus` StrEnum with four states — added to `_engine_solve.py`, exported from `math/__init__.py`
- **1c** `warmup_state()` decomposed; `_replay_profit_variance` and `_scale_to_cash` extracted
- **2b** `CONTRIBUTING.md` — covers dev setup, Conventional Commits, engine mixin pattern, PR checklist
- **2c (partial)** `bug_report.yml` and `feature_request.yml` issue templates added
- **5a** Benchmark regression gate — `rhiza_benchmarks.yml` posts PR comments on >200% regression, fails on `main` pushes; `BENCHMARKS.md` documents baseline process
- **5b** License compliance scan — `rhiza_licenses.yml` added (16th workflow)
- **9a (partial)** `CODE_OF_CONDUCT.md` (Contributor Covenant), `CHANGELOG.md` (Keep a Changelog, 5 versions), `DISCUSSION_TEMPLATE/q-and-a.yml`, `dependabot.yml`, `secret_scanning.yml` all added

---

## Remaining Work

---

### 1. Code Quality: 8 → 10

#### 1b. Eliminate remaining `_iter_solve` branch duplication  _(~40 lines)_

`_compute_mask`, `_check_signal`, and `_scale_to_cash` helpers exist but the EWMA and
sliding-window branches of `_iter_solve` still duplicate ~40 lines of mask/signal/exception
logic inline. Extract a shared position-computation helper:

```python
def _compute_position(
    self,
    mask: np.ndarray,
    expected_mu: np.ndarray,
    matrix: np.ndarray,
) -> tuple[np.ndarray, SolveStatus]:
    """Shared solve step used by both covariance branches."""
    ...
```

Both branches then call `_compute_position` and only differ in how `matrix` is produced,
which is already isolated in `_iter_matrices`. This reduces `_iter_solve` from ~100 lines
to ~60 and eliminates the duplicated exception handling and denominator checks.

**Acceptance criteria:** `_iter_solve` ≤ 60 lines. No block of >3 lines appears in both branches.

#### 4a. Add `TypeAlias` annotations for generator yield types  _(30 min)_

The generators in `_SolveMixin` yield untyped tuples. Add to `_engine_solve.py`:

```python
from typing import TypeAlias

MatrixYield: TypeAlias = tuple[int, np.ndarray | None, np.ndarray]
SolveYield:  TypeAlias = tuple[int, datetime.date, np.ndarray, np.ndarray, SolveStatus]
```

Annotate `_iter_matrices` as `Generator[MatrixYield, None, None]` and `_iter_solve` as
`Generator[SolveYield, None, None]`. Eliminates the doc-vs-code mismatch currently
flagged by `ty`.

---

### 2. Documentation: 9 → 10

#### 2a. Deploy MkDocs Material to GitHub Pages

The docs directory already has `ARCHITECTURE.md`, `GLOSSARY.md`, `SECURITY.md`,
`QUICK_REFERENCE.md`, `BENCHMARKS.md`, and ADRs. Wire them into a `mkdocs.yml`:

```yaml
site_name: basanos
theme:
  name: material
  features: [navigation.tabs, content.code.copy]
plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true
```

Add a `rhiza_docs.yml` GitHub Actions workflow that builds and deploys to `gh-pages` on
every release tag. Add a docs badge to the README.

#### 2c (remaining). Add PR template

`.github/PULL_REQUEST_TEMPLATE.md` — checklist: tests added, docstring updated,
`CHANGELOG.md` entry, `make fmt` run.

---

### 3. Testing: 8 → 10

#### 3a. Add mutation testing with mutmut

```toml
# pyproject.toml
[tool.mutmut]
paths_to_mutate = "src/basanos/math/"
tests_dir = "tests/test_math/"
```

Add `rhiza_mutation.yml` workflow that runs `mutmut run` on PRs touching
`src/basanos/math/` and fails if mutation score drops below 80%. Catches bugs that
property-based tests miss: off-by-one in EWM decay factor, wrong sign in shrinkage
blend, flipped inequality in denominator guard.

#### 3b. Add numerical stability test suite

New file `tests/test_math/test_numerical_stability.py`:

```python
# Near-singular correlation matrices (condition number > 1e12)
# min_corr_denom boundary: inputs that produce denominator exactly at 1e-14
# Ill-conditioned shrinkage (shrink=0.0 with highly collinear assets)
# Verify IllConditionedMatrixWarning fires at the documented threshold
# Assert SolveStatus.DEGENERATE (not silent NaN) when matrix is singular
```

Add a Hypothesis test verifying that for any `SolveStatus.VALID` step, the residual
`‖C·x − μ‖₂` is bounded below the `solver_residual` tolerance.

---

### 6. Mathematical Rigor: 8 → 10

#### 6a. Justify `min_corr_denom = 1e-14`

Add a comment block in `_ewm_corr.py`:

```python
# min_corr_denom = 1e-14
# Rationale: float64 machine epsilon is ~2.2e-16. The EWM variance denominator
# approaches zero when com is large and the series has a long NaN run. A threshold
# of 1e-14 (~100x epsilon) avoids division by subnormal floats while remaining
# well below any economically meaningful variance (~1e-8 for daily returns).
# Empirical lower bound logged by the benchmark suite:
#   benchmarks/results/ewm_denom_distribution.json
```

Add a benchmark pass that logs the minimum observed denominator across the test dataset
to `benchmarks/results/ewm_denom_distribution.json`. Converts the magic constant into a
documented, empirically grounded value.

#### 6b. Large-universe warning and guidance

Add a `ResourceWarning` in `_validate_inputs()`:

```python
if n_assets > 150:
    warnings.warn(
        f"n_assets={n_assets} may require >{_estimate_peak_gb(n_assets, T):.0f} GB RAM. "
        "Consider SlidingWindowConfig for large universes.",
        ResourceWarning,
        stacklevel=2,
    )
```

Add a **"Scaling beyond 150 assets"** section to `ARCHITECTURE.md` documenting:
- The O(T·N²) peak-memory formula
- `SlidingWindowConfig` with small `k` (truncated SVD) as the low-memory path for N > 200
- Ledoit-Wolf analytical shrinkage (available in `scipy`) as a drop-in for very large N

---

### 7. Dependencies & Packaging: 9 → 10

#### 7a. Publish to conda-forge

Submit a conda-forge feedstock PR. Pure Python with all deps already on conda-forge —
recipe is straightforward:

```yaml
# meta.yaml
requirements:
  run:
    - python >=3.11
    - jinja2 >=3.1
    - numpy >=2.4,<3
    - plotly >=6.6.0,<7
    - polars >=1.37.1
    - pydantic >=2.12.5,<3
    - scipy >=1.17.0,<2
```

Unblocks institutional conda-only users. Add conda-forge badge to README once merged.

---

### 8. Security: 9 → 10

#### 8a. Add Semgrep with `p/numpy` ruleset

Bandit misses NumPy-specific patterns. Add `rhiza_semgrep.yml`:

```yaml
- uses: returntocorp/semgrep-action@v1
  with:
    config: >-
      p/python
      p/secrets
      p/numpy
```

`p/numpy` catches unsafe `np.loads`, silent broadcasting bugs, and dtype coercion issues
directly relevant to this codebase.

---

### 9. Community & Maintenance: 7 → 10

#### 9a (remaining). Finish contribution infrastructure

- **README badges** — CI status, coverage %, PyPI version, conda-forge version, Python versions
- **Repo topics** — add via GitHub Settings: `portfolio-optimization`, `quantitative-finance`, `correlation`, `risk-management`, `python`
- **Good First Issues** — seed 2–3 issues (e.g., "Add conda-forge recipe", "Add `__repr__` to `WarmupState`", "Add large-universe notebook") to lower the contribution barrier

#### 9b. Discoverability (one-time effort)

- Submit to the **Awesome Quant** GitHub list
- Post to **r/algotrading** and **r/quant** with a minimal worked example
- Write a short post on the C·x = μ intuition — a Marimo notebook exported to HTML and
  hosted under the GitHub Pages docs site works well for this
- Cross-post to **Quantopian community**, **QuantLib forum**, or similar

#### 9c. Ongoing cadence

- Tag a release on every meaningful merge — `CHANGELOG.md` habit already established, just needs to be kept current
- Respond to issues within 48 hours (already committed to in `SECURITY.md`)

---

### 10. Performance: 8 → 10

#### 10b. Streaming/chunked guide for large N

Add a `BasanosStream` usage guide showing how to process assets in groups when N > 150:

```python
# Recommended pattern for N > 150
for chunk in asset_chunks(prices, chunk_size=100):
    stream = BasanosStream(cfg=cfg)
    for step in stream.iter_steps(chunk, mu_chunk):
        ...
```

Document in `ARCHITECTURE.md` under the "Scaling beyond 150 assets" section (§6b) and
add a notebook to `book/marimo/notebooks/large_universe.py`.

#### 10c. Optional Numba acceleration for EWM hot loop

`scipy.signal.lfilter` processes N² IIR filters (22,500 for N=150). Add an optional
`numba` path behind a graceful import:

```python
# src/basanos/math/_ewm_corr.py
try:
    from ._ewm_corr_numba import _ewm_iir_numba as _ewm_iir
except ImportError:
    _ewm_iir = _ewm_iir_scipy  # default, no hard dependency
```

Add optional extra to `pyproject.toml`:

```toml
[project.optional-dependencies]
fast = ["numba>=0.60"]
```

Add a benchmark comparing `pip install basanos[fast]` vs base install to the results
directory. Document in `BENCHMARKS.md`.

---

## Remaining Execution Sequence

```
Phase A — Code health (half day)
  1b  Eliminate _iter_solve branch duplication via _compute_position helper
  4a  TypeAlias annotations for MatrixYield / SolveYield

Phase B — Testing completeness (2-3 days)
  3b  Numerical stability test suite
  3a  mutmut setup + rhiza_mutation.yml CI workflow
  6a  min_corr_denom justification + ewm_denom_distribution.json benchmark

Phase C — Documentation site (1 day)
  2a  MkDocs Material + mkdocstrings
  2a  rhiza_docs.yml GitHub Pages deployment workflow
  2c  PR template

Phase D — Mathematical & performance depth (2-3 days)
  6b  Large-universe ResourceWarning + ARCHITECTURE.md scaling section
  10b Streaming guide + large_universe.py notebook

Phase E — Security & distribution (1 day + conda-forge review lag)
  8a  Semgrep rhiza_semgrep.yml workflow
  7a  conda-forge feedstock submission

Phase F — Optional performance (1-2 days)
  10c Optional Numba extra + benchmark

Phase G — Community (ongoing)
  9a  README badges + repo topics + good-first-issues
  9b  Submit to Awesome Quant, community posts
  9c  Maintain release cadence
```

---

## Projected Scores

| Category | Current | After Ph A | After Ph B | After Ph C | After Ph E | After Ph G |
|---|---|---|---|---|---|---|
| Code Quality | 8 | **10** | 10 | 10 | 10 | 10 |
| Documentation | 9 | 9 | 9 | **10** | 10 | 10 |
| Testing | 8 | 8 | **10** | 10 | 10 | 10 |
| Architecture & Design | 9 | **10** | 10 | 10 | 10 | 10 |
| CI/CD & DevOps | **10** | 10 | 10 | 10 | 10 | 10 |
| Mathematical Rigor | 8 | 8 | **10** | 10 | 10 | 10 |
| Dependencies & Packaging | 9 | 9 | 9 | 9 | **10** | 10 |
| Security | 9 | 9 | 9 | 9 | **10** | 10 |
| Community & Maintenance | 7 | 7 | 7 | 7 | 7 | **10** |
| Performance | 8 | 8 | 8 | 8 | 8 | **10** |
| **Average** | **8.5** | **9.0** | **9.6** | **9.7** | **9.9** | **10.0** |
