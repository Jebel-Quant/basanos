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

## 2026-03-20 — Analysis Entry

### Summary

Since the previous entry (2026-03-19), the repository has undergone active feature consolidation around the sliding-window covariance mode. The configuration API was refactored to use a Pydantic discriminated union (`covariance_config: EwmaShrinkConfig | SlidingWindowConfig`), a latent bug in the ewma_shrink path was found and fixed via Hypothesis property testing, and test coverage was pushed from 99% to 100% (555 tests, 1,567 statements). The codebase continues to demonstrate high discipline; this entry focuses on what has changed and what remains structurally weak.

### Strengths

- **Discriminated union configuration (PR #229)**: Replacing flat `covariance_mode/n_factors/window` kwargs directly on `BasanosConfig` with `covariance_config=SlidingWindowConfig(window=W, n_factors=k)` is a material design improvement. Pydantic's discriminator field enforces that `window` and `n_factors` are only present when the mode requires them, eliminating silent misconfiguration. The pattern is cleanly extensible to future covariance modes.
- **Hypothesis found a real bug**: `inv_a_norm` raises `SingularMatrixError` for singular correlation matrices (e.g., two assets with perfectly identical prices and `shrink=1.0`), but the ewma_shrink path in `cash_position` had no handler for it. Hypothesis's property test `test_position_leverage_is_non_negative` generated the exact falsifying example that exposed this. The fix (catch `SingularMatrixError`, treat as `denom=nan`, zero positions) is minimal and correct. Property-based testing earning its cost.
- **Exception narrowing (PR #227)**: Broad `except Exception` replaced with `except (np.linalg.LinAlgError, ValueError)` in `_iter_matrices` and the sliding-window `cash_position` loop. The correct exception types are now explicit and the error surface is documented.
- **`_iter_matrices` abstraction**: The private generator yielding `(i, t, mask, matrix)` unifies diagnostic computation (`condition_number`, `effective_rank`, `solver_residual`, `signal_utilisation`) across both covariance modes. Each diagnostic now iterates a single unified sequence rather than branching independently per mode. Clean.
- **Warm-up gap detection**: The `__post_init__` guard warning when `n_rows < 2 * window` (`optimizer.py:728`) proactively catches the most common misuse of the sliding-window mode before any computation runs. The warning message is specific enough to act on.
- **Paper workflow restored**: `make paper` target exists and `paper.yml` CI workflow compiles LaTeX. The previously noted weakness (no PDF build automation) is resolved.
- **Test suite growth**: Tests grew from ~7,200 to ~8,240 lines. The new `TestSlidingWindowErrorPaths` class covers six distinct failure modes in the sliding-window path that were previously untested: SVD failure in `_iter_matrices`, SVD failure in `cash_position`, Woodbury solve failure, zero-mu short-circuit, and `SingularMatrixError` from `inv_a_norm`.

### Weaknesses

- **BENCHMARKS.md version mismatch persists**: The environment table still records `basanos version: 0.3.0` (commit `86fb095`) despite current version being `0.4.2`. Sliding-window benchmark cases were added (PR #226) but the header was not updated. A reader cannot determine whether those numbers were produced at v0.3.0 or v0.4.x.
- **`np.linalg.inv` in the Woodbury path**: `_factor_model.py:217` computes `mid = np.linalg.inv(self.factor_covariance) + ...`, explicitly inverting the factor covariance matrix. For typical `k=2..5` this is inconsequential, but it is numerically inferior to a solve formulation, inconsistent with the Cholesky-first discipline applied everywhere else in `_linalg.py`, and risks accumulating float error as `k` grows.
- **Warm-up rows indistinguishable in output**: `cash_position` returns NaN for warm-up rows, zero for zero-signal rows, and zero for degenerate-covariance rows. Downstream consumers cannot distinguish these three semantically distinct states without inspecting the mu frame and configuration themselves.
- **`pragma: no cover` on dead-code branches**: Two `k_eff < 1` branches in `_iter_matrices` and `cash_position` (`optimizer.py:898,1029`) are annotated `# pragma: no cover` rather than removed. Both are provably unreachable given validated config (`window >= 1`, `n_factors >= 1`) and the `mask.any()` precondition. Dead code should be deleted, not hidden from coverage reporting.
- **Single ADR**: `docs/adr/` contains one ADR (Rhiza framework dependency). Given non-trivial decisions made recently — discriminated union config, Woodbury identity vs. Cholesky for factor models, warm-up semantics — the ADR practice is significantly under-utilised.

### Risks / Technical Debt

- **Breaking API change with real migration surface**: PR #229 changed `BasanosConfig` construction from `covariance_mode="sliding_window", n_factors=2, window=30` to `covariance_config=SlidingWindowConfig(window=30, n_factors=2)`. Any downstream consumer using the old API fails at runtime with a Pydantic `extra_forbidden` error. The in-flight branch `copilot/document-warm-up-gap-in-basanosconfig` required two test fixes to migrate, confirming the migration surface is real.
- **Hypothesis counterexample not pinned deterministically**: The singular-matrix edge case that broke `test_position_leverage_is_non_negative` lives in `.hypothesis/examples/`. On a fresh CI runner without the database, the property test may not rediscover it for many runs. The edge case (singular shrunk correlation matrix) should be promoted to a deterministic unit test, independent of Hypothesis replay.
- **No streaming or incremental API**: `cash_position` materialises the full `(T, N)` result array before returning. Live systems appending one row at a time must re-run the full computation. This is an architectural gap that will require significant refactoring to address.
- **Polars upper bound maintenance burden**: `polars>=1.37.1,<2` requires manual bumps as new minor versions release. No compatibility shim or API deprecation test exists.

### Score

**9/10** — Unchanged from previous entry

**Rationale**: The recent changes are net positive: the discriminated union config is a genuine design improvement, the Hypothesis-found bug fix demonstrates the testing infrastructure working as intended, and exception narrowing improves debuggability. The score does not increase because the weaknesses are real (stale benchmark metadata, `np.linalg.inv` in the Woodbury path, dead-code pragma annotations, thin ADR coverage) and the breaking API change on an in-flight branch introduces migration risk. The project remains exemplary for its size and domain.

