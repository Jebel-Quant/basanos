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
