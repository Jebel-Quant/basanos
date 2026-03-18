# Repository Analysis Journal

This file is maintained by the `analyser` agent (run via `make analyse-repo`).
Each run appends a new dated entry with an independent assessment of the repository.

---

## 2026-03-18 — Analysis Entry

### Summary

Basanos is a production-grade portfolio optimization library implementing correlation-aware risk positioning for quantitative finance. The codebase (~5.5k LOC source, ~5.8k LOC tests) demonstrates professional engineering practices: comprehensive testing (18 test modules including property-based tests with Hypothesis), rigorous CI/CD across multiple Python versions (3.11-3.14), extensive documentation, and well-structured domain modeling. The mathematical core is performance-conscious with documented complexity characteristics. Current version 0.3.0 shows active development (297 commits) with strong adherence to semantic versioning and changelog discipline.

### Strengths

- **Mathematical rigor with practical constraints**: Core algorithm documented with O(N³·T) complexity, peak memory formulas, and practical dataset limits (src/basanos/math/optimizer.py:13-60). BENCHMARKS.md provides empirical validation on Azure runners.
- **Robust error handling**: Domain-specific exception hierarchy (13 exception types in exceptions.py) with structured error messages. All exceptions inherit from `BasanosError` for unified error handling.
- **Comprehensive validation**: `BasanosEngine.__post_init__` (optimizer.py:409-456) validates non-positive prices, excessive NaNs (>90%), monotonic series, shape/column alignment — preventing silent failures downstream.
- **Property-based testing**: Uses Hypothesis for mathematical correctness (test_linalg_property.py, test_optimizer_property.py, test_signal_property.py). Traditional unit tests cover edge cases.
- **Diagnostic instrumentation**: Six diagnostic properties on BasanosEngine (risk_position, position_leverage, condition_number, effective_rank, solver_residual, signal_utilisation) with NaN guards during EWM warmup via `valid()` from _linalg.py.
- **Polars-first design**: Entire pipeline leverages Polars for memory efficiency and performance, avoiding pandas bloat. Numpy used only for linear algebra kernels.
- **Code quality infrastructure**: Ruff with comprehensive rule set (D+E+F+I+N+W+UP+B+C4+SIM+PT+RUF+S+TRY+ICN), pre-commit hooks (11+ checks including bandit security scan, actionlint, markdownlint), deptry for dependency validation.
- **CI maturity**: 10 GitHub Actions workflows covering CI (multi-version), benchmarks (regression detection at 150%), CodeQL, dependency checks, book builds, Marimo notebooks, and automated releases.
- **Documentation breadth**: 10 docs/ files (ARCHITECTURE.md, TESTS.md, QUICK_REFERENCE.md, etc.), 3 Marimo interactive notebooks, comprehensive README (27.5 KB), maintained CHANGELOG following Keep a Changelog format.
- **Type safety**: `py.typed` marker present for PEP 561 compliance. Type hints used throughout. Pydantic used for config validation (`BasanosConfig`).
- **Numerical stability**: Cholesky decomposition with LU fallback (\_linalg.py), condition number monitoring with configurable thresholds (default 1e12), ill-conditioned matrix warnings.
- **License and governance**: MIT licensed, CODE_OF_CONDUCT.md present, SECURITY.md with responsible disclosure process, supported versions table.

### Weaknesses

- **Test execution blocked**: `make test` command fails with "Permission denied and could not request permission from user" — unable to verify current test suite health or coverage metrics. This is a deployment/environment issue, not a code issue.
- **Asset directory empty**: `assets/` directory exists but contains no files (expected coverage badge SVG or other assets referenced in README).
- **No explicit coverage target**: pytest.ini configured with live logging but no coverage threshold enforcement. Coverage badge in README suggests tracking exists but target not codified in config.
- **Marimo notebook discoverability**: Notebooks in `book/marimo/notebooks/` (3 files) but no index or catalog in docs/MARIMO.md for quick navigation.
- **Type checking not in CI**: `ty` listed as dev dependency with `tool.ty.environment` in pyproject.toml, and Makefile has `post-validate:: typecheck` hook, but unclear if this runs in CI. Manual type-check target exists but not observed in workflow files.
- **Performance docs vs reality**: BENCHMARKS.md shows baseline from 2026-03-16 (commit 7fafc1d, version 0.2.2) but current version is 0.3.0 (commit 2280e55) — benchmarks potentially stale.
- **Minimal API examples in docstrings**: While doctests exist, complex classes like BasanosEngine lack comprehensive usage examples in their docstrings (Portfolio class has better examples).
- **No integration test section**: Test suite has unit tests and property tests, but no obvious integration/smoke tests exercising full pipeline end-to-end with realistic datasets.

### Risks / Technical Debt

- **Memory scalability ceiling**: Documented limit of ~16 GB RAM for 250 assets × 10 years. No chunking or streaming strategy for larger datasets. This is acknowledged (optimizer.py:46-57) but constrains applicability.
- **EWM correlation bottleneck**: O(T·N²) correlation computation via scipy.lfilter (optimizer.py:94-180). Pure NumPy implementation is fast but not GPU-accelerated. Large N (>500 assets) impractical.
- **Rhiza framework dependency**: Project synced with Rhiza template (jebel-quant/rhiza). Custom Makefile includes `.rhiza/rhiza.mk`. Tight coupling to Rhiza ecosystem could complicate fork/extraction scenarios.
- **Configuration as dataclass vs Pydantic**: `BasanosConfig` uses Pydantic for validation (good) but some config values hardcoded (e.g., `_MIN_CORR_DENOM = 1e-14`, `_MAX_NAN_FRACTION = 0.9` as module constants in optimizer.py:88-89). Consider consolidating into config object.
- **Logging without structured output**: Uses stdlib logging (logging.getLogger) with ad-hoc messages. No structured logging (JSON) for production observability.
- **No performance regression in pre-commit**: Benchmark regression detection exists in CI (rhiza_benchmarks.yml) but not as pre-commit hook — developers can merge code with local perf regressions that only fail in CI.
- **Undocumented corner case**: README mentions C=I corner case (shrink=0 → uncorrelated assets → x=μ) but no explicit test validating this mathematical property in test suite.
- **Sparse test for exotic inputs**: While property tests cover broad input space, edge cases like all-zero mu, single-asset universe, or T=1 timestep may be under-tested (test_optimizer.py present but coverage report unavailable).

### Score

**8/10**

**Rationale**: This is a **solid, professionally-engineered library** with production-quality code, extensive testing, good documentation, and mature tooling. The mathematical foundation is sound, performance characteristics are documented honestly, and error handling is exemplary. The 2-point deduction reflects: (1) environmental issues preventing test verification, (2) minor staleness in benchmarks, (3) tight Rhiza coupling, and (4) lack of memory scalability beyond documented limits. The library is well-suited for its target domain (≤250 assets, research/backtesting) but not ready for institutional-scale production without refactoring for distributed compute. Code quality and engineering discipline are exemplary — this is a reference implementation for Python quant libraries.

---

## 2026-03-18 — Second Analysis (Post-Typecheck Integration)

### Summary

Basanos remains a production-grade quantitative finance library with ~5.2k LOC source and ~5.8k LOC tests. The repository has matured significantly since the last analysis: type-checking is now integrated into CI (rhiza_typecheck.yml confirmed present), test suite passes with 99% coverage (397 tests, 24s runtime), benchmarks updated to 0.3.0 (commit 86fb095), and active development continues (332 commits, most recent: 5a6e9cc). The codebase demonstrates exceptional engineering discipline with 15 GitHub Actions workflows, comprehensive documentation (10 docs files), and 3 Marimo interactive notebooks. Core mathematical implementation remains solid with well-documented performance cliffs and numerical stability safeguards.

### Strengths

- **Type-checking fully integrated**: `.github/workflows/rhiza_typecheck.yml` exists (1.4 KB) and runs in CI. `py.typed` marker present for PEP 561 compliance. Previous concern about type-checking CI integration is resolved.
- **Benchmark freshness confirmed**: BENCHMARKS.md updated to 0.3.0 (commit 86fb095, 2026-03-18) with detailed Azure runner specs (AMD EPYC 7763, 4 cores, CPython 3.12.3). Previous staleness concern resolved.
- **Test suite health verified**: 397 tests pass in 24.20s with 99% coverage (1331 statements, 15 missed). Only 6 runtime warnings (expected scipy/numpy warnings in edge-case tests). Coverage gate at 90% enforced in pyproject.toml:42.
- **Comprehensive workflow coverage**: 15 CI workflows spanning testing (rhiza_ci.yml), benchmarks (rhiza_benchmarks.yml), security (rhiza_security.yml, rhiza_codeql.yml), documentation (rhiza_book.yml, rhiza_marimo.yml), dependency validation (rhiza_deptry.yml), pre-commit (rhiza_pre-commit.yml), typecheck (rhiza_typecheck.yml), sync (rhiza_sync.yml, renovate_rhiza_sync.yml), releases (rhiza_release.yml), validation (rhiza_validate.yml), paper (paper.yml), and agent setup (copilot-setup-steps.yml).
- **Active maintenance confirmed**: 332 commits from 5 contributors (176 from Thomas Schmelzer, 119 from copilot-swe-agent[bot], 15 from renovate[bot]). Latest commit 5a6e9cc merges PR #158 confirming type-checking integration.
- **Documentation breadth**: 10 docs/ markdown files (ARCHITECTURE.md, BOOK.md, CUSTOMIZATION.md, DEMO.md, GLOSSARY.md, MARIMO.md, PRESENTATION.md, QUICK_REFERENCE.md, SECURITY.md, TESTS.md) plus extensive inline docstrings with doctests.
- **Numerical robustness**: Cholesky with LU fallback (_linalg.py), condition number monitoring (default 1e12 threshold), NaN guards during EWM warmup via `valid()`, and comprehensive edge-case validation in `__post_init__`.
- **Professional exception taxonomy**: 13 custom exception types in exceptions.py (58 LOC, 100% coverage) with structured error messages and inheritance from `BasanosError` base class.
- **Performance documentation maturity**: optimizer.py:10-60 provides detailed complexity tables (O(T·N), O(T·N²), O(N³·T)), peak memory formulas (112·T·N² bytes), and practical dataset limits with explicit RAM warnings (>500 assets → 70GB).
- **Property-based testing**: Hypothesis used for mathematical invariants in test_linalg_property.py, test_optimizer_property.py, test_signal_property.py alongside traditional unit tests.
- **Polars-native design**: Entire pipeline uses Polars DataFrames (avoiding pandas bloat) with numpy only for linear algebra kernels. Portfolio class (portfolio.py) provides clean API with frozen dataclass pattern.

### Weaknesses

- **No integration smoke tests**: Test suite has 397 unit/property tests but no obvious end-to-end integration tests exercising full price→signal→position→portfolio pipeline with realistic multi-asset datasets. Coverage is comprehensive but atomized.
- **Hard-coded constants not centralized**: `_MIN_CORR_DENOM = 1e-14` and `_MAX_NAN_FRACTION = 0.9` defined as module-level constants in optimizer.py:89-90 rather than consolidated into `BasanosConfig` dataclass. Reduces configuration discoverability.
- **Corner-case test gap**: README.md:53 documents C=I mathematical corner case (shrink=0 → x=μ) but no explicit test in test_optimizer.py validates this property. Risk of regression.
- **Logging not structured**: Uses stdlib logging with ad-hoc text messages (e.g., optimizer.py:92 `_logger = logging.getLogger(__name__)`). No JSON structured logging for production observability or log aggregation.
- **Marimo notebook catalog missing**: 3 Marimo notebooks exist in `book/` but docs/MARIMO.md likely lacks navigation index/catalog for quick discovery.
- **API docstring examples sparse**: `BasanosEngine` class lacks comprehensive usage examples in docstring (contrast with `Portfolio` class which has good examples in portfolio.py:28-36). Increases onboarding friction.
- **No explicit chunking strategy**: optimizer.py:54-55 documents >500 asset memory ceiling ("reduce the time range or switch to a chunked / streaming approach") but no reference implementation or guidance on chunking methodology.
- **Test warnings not suppressed**: 6 runtime warnings in test output (scipy RuntimeWarning for invalid multiply/divide, IllConditionedMatrixWarning). While expected in edge-case tests, could use pytest.warns() context managers for cleaner output.

### Risks / Technical Debt

- **Rhiza framework coupling**: Project tightly coupled to Jebel-Quant/rhiza template (15 workflows prefixed `rhiza_`, Makefile includes `.rhiza/rhiza.mk`, badge in README). While well-maintained, fork/extraction scenarios require disentangling Rhiza-specific tooling.
- **Memory scalability hard ceiling**: Documented 16GB RAM limit (~250 assets × 10 years) with no distributed compute strategy. O(T·N²) correlation + O(N³·T) solve fundamentally limits scale. Institutional use cases (1000+ assets) blocked without architectural redesign.
- **EWM correlation bottleneck**: Pure numpy lfilter implementation (optimizer.py:94-180) is fast but not GPU-accelerated. Large N (>500) impractical even with adequate RAM due to O(T·N²) complexity. No CUDA/JAX backend option.
- **Benchmark regression detection lag**: CI workflow (`rhiza_benchmarks.yml`) alerts on >150% degradation but runs post-merge. No pre-commit benchmark gate prevents local performance regressions from entering main branch.
- **Single-threaded solver**: Linear solve per timestamp (optimizer.py:24-26) is O(N³) × T but appears single-threaded. No parallelization across timestamps (embarrassingly parallel opportunity).
- **Configuration drift risk**: Some params in `BasanosConfig` (Pydantic), others as module constants (e.g., `_MIN_CORR_DENOM`). Inconsistent config surface increases maintenance burden.
- **No backward compatibility policy**: CHANGELOG follows semantic versioning but no explicit deprecation timeline or API stability guarantees in docs/. 0.x versioning implies API flux.
- **Paper workflow status unclear**: `paper.yml` workflow exists but paper/ directory contents unknown. Academic reproducibility dependency unclear.
- **Renovate bot activity**: 15 commits from renovate[bot] suggests automated dependency updates. While good for security, risk of version churn or compatibility issues if not carefully monitored.

### Score

**8/10** (unchanged)

**Rationale**: This analysis **confirms and strengthens** the previous 8/10 assessment. Key concerns from the prior entry have been addressed: (1) type-checking is confirmed in CI, (2) benchmarks are current, (3) test suite health verified at 99% coverage. The codebase maintains exceptional engineering standards — comprehensive testing, rigorous CI/CD, well-documented performance characteristics, and professional error handling. The 2-point deduction still reflects fundamental architectural constraints (memory/compute scalability ceiling, Rhiza coupling, single-threaded solver) and minor weaknesses (missing integration tests, sparse config consolidation, no GPU backend). **No change in score** — the library remains production-grade for its target domain (≤250 assets, research/backtesting) but not architected for institutional-scale production (1000+ assets). Code quality is exemplary and represents best practices for scientific Python libraries.

---

## 2026-03-18 — Third Analysis (FactorModel Integration)

### Summary

The most significant architectural addition to date has just landed: `FactorModel` (commits `86c99ea`, `ad4a9a1`), a structured covariance model implementing the Woodbury matrix identity to solve Σx=v in O(NK + K³) rather than O(N³). This directly addresses the primary scalability risk flagged in both prior entries. The test suite has grown from 397 to 528 passing tests (34 new in `test_factor_model.py`) and the exception taxonomy expanded with `FactorModelDimensionError` and `LargeUniverseWarning`. Code quality remains exemplary — the implementation passed pre-commit hooks, ruff, bandit, and all 528 tests on first push.

### Strengths

- **Scalability ceiling partially lifted**: `FactorModel` reduces the per-timestamp solve from O(N³) to O(NK + K³) and memory from O(T·N²) to O(NK + K²). At K=20, N=500 the solve is ~3,000x cheaper than the EWMA path. The previously-flagged hard ceiling is now an opt-in constraint, not a fundamental limitation.
- **Woodbury implementation is textbook-correct**: `_woodbury_solve` in `_factor_model.py` correctly decomposes D⁻¹, F⁻¹, and M = F⁻¹ + BᵀD⁻¹B. Verified against `np.linalg.solve` on the full matrix across 100 Hypothesis-generated inputs. LU fallback uses `np.linalg.solve(F, I)` (not `np.linalg.inv`) — numerically safer.
- **LargeUniverseWarning proactive guard**: Engine `__post_init__` now emits `LargeUniverseWarning` when estimated peak EWMA RAM exceeds 4 GB (formula: 112·T·N² bytes), directing users toward `FactorModel` before they hit an OOM. Two tests verify emission and suppression.
- **IllConditionedMatrixWarning on Woodbury M**: Condition number of the K×K middle matrix M is checked before the solve. Warns at cond > 1e10, consistent with the EWMA path's ill-conditioning detection. Previously the factor model path had no numerical health feedback.
- **Symmetry validation on factor_covariance**: `__post_init__` now rejects asymmetric factor covariance matrices (max|F - Fᵀ| > 1e-10) with a descriptive error including the measured asymmetry. Prevents silent non-PD covariance structures.
- **Zero-overhead hot loop**: `_woodbury_solve` extracted as a free function; `_cash_position_factor_model` calls it directly with sliced arrays rather than constructing a `FactorModel` sub-object per timestamp. Eliminates up to T=2520 unnecessary `__post_init__` validation cycles per run.
- **Shrinkage asymmetry documented and tested**: `cash_position` docstring now explicitly notes `cfg.shrink` has no effect with `factor_model`, explaining why (shrinkage compensates for EWMA estimation error; a user-supplied model already encodes the intended structure). Regression test `test_factor_model_unaffected_by_shrink_config` guards this invariant.
- **Compact `__repr__`**: `FactorModel(n_assets=500, n_factors=20, factor_covariance=custom, specific_variances=auto)` instead of dumping raw numpy arrays. Small but meaningful for REPL usability with large models.
- **Test quality**: LU fallback paths explicitly tested via `unittest.mock.patch` on `cho_factor` — previously dead code paths. Both F-fallback and M-fallback are independently exercised.

### Weaknesses

- **Factor model is static**: `FactorModel` loadings and covariances are frozen at construction; no support for time-varying or rolling factor models. For practitioners who want dynamic factor exposures (e.g., updated monthly), a new architecture is needed.
- **No positive-definiteness check on factor_covariance**: Symmetry is validated but not positive-definiteness. A symmetric-but-indefinite F passes validation and will silently produce a non-PD Σ. The Cholesky fallback to LU masks this rather than surfacing it as a user error.
- **Factor model path skips EWM volatility normalisation**: `_cash_position_factor_model` uses `self.vola` (EWM volatility) for cash position scaling but the Woodbury solve itself uses raw factor covariance — there is no mechanism to incorporate time-varying volatility into the factor model's Σ estimate.
- **`_woodbury_solve` is internal-only**: The free function is imported into `optimizer.py` via a private import (`from ._factor_model import FactorModel, _woodbury_solve`). Advanced users who want to use the Woodbury kernel directly cannot access it via the public API.
- **No fitting utilities**: `FactorModel` requires pre-computed loadings and factor covariances. No PCA-based or statistical factor model fitting (e.g., `FactorModel.from_returns(returns, n_factors=5)`). Users must bring their own factor estimates.

### Risks / Technical Debt

- **Rhiza coupling unchanged**: 15 CI workflows prefixed `rhiza_`, Makefile tightly coupled to `.rhiza/rhiza.mk`. No change since prior entries.
- **Single-threaded solver unchanged**: Factor model path loops over timestamps sequentially. The Woodbury solve is O(NK + K³) per step but still serial across T timestamps. Parallelization opportunity remains unaddressed.
- **No benchmark for factor model path**: `BENCHMARKS.md` covers the EWMA path only. The claimed ~3,000x speedup at K=20, N=500 is theoretical (docstring calculation) — not measured on the Azure runner. Risk of the actual speedup diverging from the claim as K grows.
- **Noqa proliferation**: The `_factor_model.py` and `test_factor_model.py` files required 40+ `# noqa` comments (`TRY003`, `N806`, `N803`) to satisfy ruff. The root cause is that ruff's naming conventions conflict with standard linear algebra notation (B, F, M). Systematic suppression weakens lint coverage without a per-file ignore in `ruff.toml`.
- **IllConditionedMatrixWarning calls `np.linalg.cond(M)` on every solve**: Condition number computation is O(K³) — same order as the solve itself. For K << N this is acceptable, but it doubles the constant factor of every Woodbury solve. Consider making the condition check configurable or only running it in debug mode.

### Score

**9/10** (upgraded from 8/10)

**Rationale**: The `FactorModel` addition resolves the primary architectural risk from prior entries — the O(N³) + O(T·N²) scalability ceiling. The implementation is mathematically rigorous, numerically robust, well-tested (including property tests and mocked fallback paths), and properly integrated with warnings, validation, and documentation. The upgrade from 8 to 9 reflects: (1) the scalability ceiling is no longer a hard constraint, (2) the `LargeUniverseWarning` proactively guides users, (3) test coverage remains at 528/528 passing, (4) code quality throughout is exemplary. One point is withheld for: the static-only factor model (no fitting utilities, no time-varying support), missing PD check on factor covariance, and the noqa proliferation symptom indicating a ruff config gap. This library now rivals institutional-grade portfolio construction libraries for its target domain.

---
