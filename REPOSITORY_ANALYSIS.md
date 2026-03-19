# Basanos — Repository Analysis Journal

**Project**: Basanos — Correlation-aware portfolio optimization and analytics for Python  
**Repository**: https://github.com/jebel-quant/basanos  
**Purpose**: Ongoing technical review of code quality, architecture, and maintainability

---

## 2026-03-19 — Initial Analysis Entry

### Summary

Basanos is a mature, production-grade Python library implementing correlation-adjusted risk position optimization for systematic trading strategies. The codebase exhibits exceptional quality: 100% test coverage across 1,474 statements, rigorous type safety, clean architecture, comprehensive documentation, and professional CI/CD infrastructure. The mathematical core is well-documented with performance characteristics explicitly stated. The project uses the rhiza framework for standardized Python development workflows and demonstrates outstanding engineering discipline across all dimensions.

### Strengths

- **Test coverage**: Perfect 100% coverage (1,474/1,474 statements, 519 passing tests, 22 test files, ~7,200 lines of test code vs ~5,600 lines of source). Property-based testing with Hypothesis included.
- **Code organization**: Clean separation of concerns: `math/` (optimizer, factor models, linalg), `analytics/` (portfolio stats, reports, plots), clear exception hierarchy with 14 custom domain exceptions (`exceptions.py`).
- **Mathematical rigor**: Performance complexity documented explicitly in `optimizer.py:1-61` (O(T·N²) correlation, O(N³·T) solving, peak memory: 112·T·N² bytes). Practical limits stated (≤150 assets/5yr on 8GB, ≤250 assets/10yr on 16GB).
- **Numerical stability**: `_linalg.py` implements Cholesky-first solving with LU fallback, condition number monitoring (threshold: 1e12), explicit positive-definite checks, and `IllConditionedMatrixWarning` for bad matrices.
- **Type safety**: `py.typed` marker present, `ty` type-checker integrated in CI (`rhiza_typecheck.yml`), pydantic for configuration validation (`BasanosConfig`).
- **Documentation depth**: README (33.9KB), dedicated problem statement (`problem.md`), architecture diagrams (`docs/ARCHITECTURE.md`), performance benchmarks (`BENCHMARKS.md` with pytest-benchmark CI regression detection at 150% threshold), eight docs files in `docs/`.
- **CI/CD maturity**: 14 GitHub Actions workflows covering: CI, pre-commit, type-checking, deptry, CodeQL, security scanning, automated releases, documentation builds, benchmark regression, renovate sync. Copilot agent hooks (`copilot-setup-steps.yml`, `.github/hooks/hooks.json`).
- **Dependency hygiene**: Zero issues from `deptry`, explicit `package_module_name_map` in `pyproject.toml`, minimal runtime dependencies (numpy, scipy, polars, plotly, pydantic), dev dependencies properly separated.
- **Version control**: 409 commits, active development (20 commits visible in recent log), semantic versioning (v0.4.2), detailed changelog following keepachangelog.com format.
- **Code style**: Ruff-formatted, pre-commit hooks enforced, editorconfig present, no dead code or commented-out blocks observed.
- **Error handling**: Comprehensive `raise ... from exc` pattern throughout to preserve stack traces (noted in CHANGELOG.md as a deliberate fix). Input validation in `__post_init__` methods (e.g., `Portfolio`, `BasanosEngine`).
- **Academic grounding**: LaTeX paper source in `paper/basanos.tex` (53KB), bibliography (`basanos.bib`), badge linking to paper in README. Demonstrates intent for academic peer review.

### Weaknesses

- **Factor model integration**: `FactorModel` frozen dataclass exists (`_factor_model.py`) with `from_returns()` factory and `solve()` method, but integration with main optimizer appears recent (memories reference rolling windows, PCA covariance modes). Test coverage exists but usage patterns across the codebase need verification to ensure feature is fully baked.
- **Paper artifact missing**: README links to `paper/basanos.pdf` but only `.tex` and `.bib` files exist. Badge implies published paper but PDF is absent from repository.
- **Performance documentation lag**: BENCHMARKS.md environment shows "basanos version: 0.3.0" (commit `86fb095`) but current version is 0.4.2, suggesting baseline results are two minor versions behind.
- **README size**: 33.9KB README is too large to read at once via tooling. Consider splitting into modular docs (quickstart, concepts, performance, configuration) with README as navigation hub.
- **Memory profiling gap**: While computational complexity is exhaustively documented, actual memory profiling results (not just theoretical calculations) are not visible. Would benefit from `memory_profiler` or `tracemalloc` data for validation.
- **Minimal API surface**: Only 14 Python files in `src/`, which is excellent for complexity but raises question of whether certain common use cases (e.g., factor orthogonalization, portfolio rebalancing, transaction cost modeling) require external glue code.

### Risks / Technical Debt

- **Scalability ceiling**: Documented limits (>500 assets infeasible on 16GB RAM) make this unsuitable for high-frequency or large-universe applications without chunking strategy. This is acknowledged but no streaming/chunked approach is roadmapped.
- **EWM correlation memory**: `_ewm_corr_numpy()` allocates 14 float64 arrays of shape (T,N,N) simultaneously. For 1000 assets × 2520 days this is ~70GB. No memory-efficient alternative (e.g., iterative/rolling) is provided for users exceeding hardware limits.
- **Sparse matrix opportunity**: Correlation matrices often have block structure or sparsity (sector clusters, factor exposure). No exploitation of sparse solvers (`scipy.sparse.linalg`) or block-diagonal structure is evident, potentially leaving 2-10× speedup on table for structured universes.
- **PCA covariance mode validation**: Repository memories indicate `CovarianceMode.pca` was recently added with `pca_components` config field and `sharpe_at_pca_components()` diagnostic. This suggests bleeding-edge feature - needs time in production to prove stability.
- **Polars version pinning**: `polars>=1.37.1,<2` is tight pinning. Polars is pre-1.0 and has history of breaking changes. May require frequent maintenance releases.
- **Rhiza framework dependency**: Project is "synced with rhiza" per badge. If rhiza templates change substantially or are deprecated, substantial makefile/CI rework may be required. Coupling is high (14 CI workflows, all `rhiza_*` prefixed).
- **Academic paper workflow**: `.tex` file present but no compilation workflow (no `make paper`, no paper build in CI). Makes paper maintenance error-prone and risks divergence from code.

### Score

**9/10** — Exemplary, production-grade, rigorously maintained

**Rationale**:
This is a textbook example of how to structure a scientific Python library. The codebase exhibits professional discipline in every dimension: tests, types, documentation, error handling, CI/CD, dependency management, and performance transparency. The mathematical implementation is numerically sound (Cholesky-first solving, condition monitoring) and performance characteristics are documented with refreshing honesty. The few weaknesses identified are minor (stale benchmark baseline, missing compiled paper) or inherent to the problem domain (memory scaling). The only reason this is not a 10/10 is the nascent state of the factor model features (incomplete integration validation) and the absence of sparse matrix optimization for structured portfolios. For a v0.4.2 library, this is outstanding work.

---

## 2026-03-19 — Analysis Entry #2

### Summary

Three PRs merged today address three of the six weaknesses and one of the six risks flagged in the initial entry. The headline change is `sliding_window` covariance mode (PR #214): a rolling-window factor model integrated directly into `BasanosEngine`, implementing Section 4.4 of the paper. The optimizer source grew from ~1,079 to 1,555 lines and its test file from ~1,789 to 2,090 lines, maintaining the 100% coverage invariant. The paper compilation gap is closed (`make paper`). The Rhiza coupling risk is now formally documented with a tested exit strategy (ADR 0001).

### Strengths

- **Sliding window mode is properly engineered**: `CovarianceMode` (StrEnum, `optimizer.py:209`) dispatches cleanly inside `_iter_matrices()` and `cash_position`. The `ewma_shrink` path is unchanged. The new path caps `k_eff = min(win_k, win_w, n_sub)` defensively and falls through to `None` on any exception — correct fail-safe behaviour for production use.
- **Woodbury solve integrated cleanly**: `FactorModel` (`_factor_model.py`) is now imported and used by the optimizer. Its O(k³+kN) solve path is documented in both the class docstring and `BasanosConfig` field description. The public API exports `CovarianceMode` from `basanos.math.__init__` — users don't need to know where it lives.
- **`model_validator` enforces cross-field constraints**: `_validate_sliding_window_params` (`optimizer.py:546`) raises at construction time if `window` or `n_factors` are missing in sliding-window mode. Pydantic's `mode="after"` is the correct hook for cross-field validation.
- **`typing.cast` replaces assertions in `src/`**: Code review applied on PR #214 (`b866177`). `cast(int, self.cfg.window)` / `cast(int, self.cfg.n_factors)` at `optimizer.py:803-804` satisfy the type-checker without runtime overhead and without violating the no-assert-in-src convention.
- **Paper weakness closed**: `make paper` target (`Makefile:25-31`) guards against missing `latexmk`, prints clear error, and compiles `paper/basanos.tex → paper/basanos.pdf`. The "no paper compilation workflow" weakness from the initial analysis is resolved.
- **ADR practice adopted**: `docs/adr/0001-rhiza-framework-dependency.md` demonstrates architectural decision logging. The ADR correctly notes that all rhiza files are committed (no runtime fetch), that sync is opt-in via a pinned tag (`v0.8.14`), and provides a four-step exit strategy. Risk now documented and mitigated.

### Weaknesses

- **Silent `except Exception` in `_iter_matrices`** (`optimizer.py:821-822`): Any failure in `FactorModel.from_returns()` — including programming errors, not just numerical ones — silently yields `None` (zero position). This is conservative but makes debugging harder. A narrower except clause (e.g., `except (np.linalg.LinAlgError, ValueError)`) with a logged warning would be safer and easier to diagnose.
- **`sliding_window` warm-up gap undocumented at API level**: The condition `i + 1 < win_w` (`optimizer.py:808`) means the first `W-1` rows always yield zero positions regardless of signal. This is correct but is not surfaced in `BasanosConfig.window`'s field description. Users setting `window=252` on a 260-row dataset will get 251 rows of zeros with no warning.
- **No benchmark baseline for `sliding_window` mode**: `BENCHMARKS.md` covers only `ewma_shrink`. The O(T·W·N·k) sliding-window path has a qualitatively different scaling curve and should have its own regression baseline, especially since the hot path calls `np.linalg.svd` per row.
- **Paper PDF still absent from repository**: `make paper` now compiles locally, but `README.md` still badges to a PDF that is not committed. The badge URL likely points to a CI artifact or external host; this should either be committed or the badge pointed at the CI-generated artifact explicitly.
- **ADR index is a single file for now**: `docs/adr/README.md` exists but with one entry. Not a weakness yet, but the ADR practice needs to be applied retroactively to prior significant decisions (e.g., choice of polars over pandas, Woodbury identity solve strategy, EWM over rolling correlation) to make the ADR log genuinely useful.

### Risks / Technical Debt

- **`sliding_window` not yet battle-tested**: The feature was implemented and merged in a single sprint. `FactorModel.from_returns()` delegates to `np.linalg.svd` on potentially ill-conditioned windows. Real-world data (NaNs filled with 0.0 at `optimizer.py:812`) could yield degenerate windows. The silent fallback masks failures — this is appropriate for production but needs monitoring hooks to detect systematic fallback rates.
- **Polars version ceiling still tight**: `polars>=1.37.1,<2` unchanged. This note stands from the initial analysis.
- **`window` and `n_factors` are `int | None` in frozen config**: The `None` default for fields that are required in one mode is an awkward pattern. If `CovarianceMode` gains a third mode that needs different required fields, this pattern becomes hard to maintain. A discriminated union (`Annotated[Union[EwmaShrinkConfig, SlidingWindowConfig], ...]`) would be cleaner long-term.

### Score

**9/10** — Score unchanged; the quality bar is maintained and previous weaknesses actively addressed

**Rationale**:
The sliding window integration is technically sound and the public API is clean. Today's PRs directly close three items from the prior weakness/risk list: missing paper compilation, undocumented Rhiza coupling, and unintegrated factor model. The remaining gap (no benchmark for the new mode, silent broad exception, warm-up gap undocumented) are minor. The codebase continues to demonstrate the same discipline as the initial assessment.

