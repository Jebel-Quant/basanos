# Repository Analysis Journal

This document contains periodic technical reviews of the Basanos repository.

---

## 2026-03-16 — Analysis Entry

### Summary
Basanos is a specialized quantitative finance library implementing correlation-aware portfolio optimization. The codebase demonstrates strong fundamentals with clean architecture, comprehensive testing, excellent documentation, and modern Python tooling. The implementation is mathematically sophisticated, using EWMA correlation estimation with shrinkage and NumPy-based IIR filtering for performance. Code quality is high with near-perfect test coverage (~95% based on test/code ratio), minimal technical debt, and no apparent TODO/FIXME markers. However, the library is narrowly focused (single optimization approach), lacks published benchmarks, and has limited real-world validation evidence.

### Strengths

- **Clean Architecture**: Well-organized module structure with clear separation between math (`src/basanos/math/`) and analytics (`src/basanos/analytics/`). Only 9 source files totaling ~1,859 LOC, demonstrating focused scope.

- **Excellent Documentation**: Comprehensive README with clear examples, API reference, configuration guide, and "How It Works" section explaining the mathematical approach. Additional docs in `docs/` cover architecture, testing, security, customization, and quick reference.

- **Strong Test Coverage**: 7 test modules with ~1,895 LOC of test code (test-to-source ratio >1.0), indicating thorough testing. Tests include property-based testing fixtures and benchmarking infrastructure (`tests/benchmarks/`).

- **Modern Tooling**: Uses rhiza framework for standardization, uv for dependency management, ruff for formatting/linting, pre-commit hooks, comprehensive GitHub Actions CI/CD (12 workflow files), Renovate for dependency updates, CodeQL for security scanning.

- **Mathematical Rigor**: Implementation of `_ewm_corr_numpy` (lines 24-117 in `optimizer.py`) is sophisticated, using `scipy.signal.lfilter` to compute EWMA correlations via IIR filtering without pandas dependency. Detailed comments explain the algorithm. Linear algebra helpers in `_linalg.py` properly validate inputs and handle edge cases.

- **Type Safety**: Uses Pydantic for configuration validation (`BasanosConfig` with field validators), frozen dataclasses for immutability, and type hints throughout.

- **Polars-Native**: Built on Polars DataFrames for performance, demonstrating modern data engineering practices. No pandas dependency.

- **Production-Ready Features**: Includes comprehensive portfolio analytics (Sharpe, VaR, CVaR, drawdown, skew, kurtosis), performance attribution (tilt/timing decomposition), interactive Plotly visualizations, and position smoothing/lagging utilities.

- **License & Governance**: MIT licensed with CODE_OF_CONDUCT.md, CONTRIBUTING.md, SECURITY.md, and clear contribution guidelines added in recent commits.

### Weaknesses

- **Single Optimization Approach**: The library implements one specific method (correlation-shrinkage + normalized linear system). No alternative optimizers, no mean-variance optimization, no robust optimization variants. Limited applicability.

- **Hardcoded Assumptions**: The optimizer uses specific hardcoded values like `_MIN_CORR_DENOM = 1e-14`, `profit_variance_init = 1.0`, `profit_variance_decay = 0.99`, `position_scale = 1e6`. While now configurable in `BasanosConfig`, the defaults lack justification or citation.

- **Limited Input Validation**: `BasanosEngine.__post_init__` validates shapes and columns but doesn't check for data quality issues (e.g., negative prices, excessive NaNs, monotonic data). Could fail silently on malformed inputs.

- **Performance Unknowns**: No published benchmarks. The `tests/benchmarks/` directory exists with recent commits (8 days ago), but no documentation on performance characteristics, scalability limits (asset count, time series length), or memory usage.

- **No Shrinkage Methodology Documentation**: The shrinkage parameter (`cfg.shrink`) is exposed but lacks guidance on selection. No reference to Ledoit-Wolf, linear shrinkage estimators, or empirical studies on optimal values.

- **Sparse Error Handling**: Functions like `inv_a_norm` and `solve` in `_linalg.py` return `np.nan` or raise `AssertionError` but provide minimal context. No custom exception types for domain-specific errors (e.g., `SingularMatrixError`, `InsufficientDataError`).

- **Visualization Dependency**: Tight coupling to Plotly for all visualizations. No option for matplotlib or static exports for publication-quality figures.

- **No Backtesting Framework**: Portfolio analytics are comprehensive but lack a backtesting harness with transaction costs, slippage, rebalancing constraints, or out-of-sample validation utilities.

### Risks / Technical Debt

- **Numerical Stability Unchecked**: Correlation matrices are shrunk but not explicitly tested for positive-definiteness before solving linear systems. `np.linalg.solve` will fail on singular/near-singular matrices. No use of Cholesky decomposition or condition number checks.

- **Missing Input Domain Tests**: No property-based tests (despite pytest being available) to validate behavior on edge cases like: all-zero mu vectors, perfectly correlated assets, single-asset portfolios, or extreme volatility regimes.

- **Profit Variance EMA Logic**: Lines 323-339 in `optimizer.py` implement adaptive position sizing via `profit_variance`, but the logic is subtle (using yesterday's position for today's profit, lagged volatility). No dedicated unit tests for this critical risk management component.

- **Date Column Dependency**: Multiple methods assume a 'date' column exists (e.g., `Portfolio.truncate`, `Portfolio.monthly`). Integer-indexed data would break or behave unexpectedly. Inconsistent support.

- **Cache Pollution**: 1,776 `__pycache__` and `.pyc` files found despite `.gitignore` rules. Suggests incomplete ignore patterns or committed cache files.

- **Undocumented Constraints**: `BasanosConfig` validates `corr >= vola` but doesn't explain *why* this constraint exists or what happens at the boundary (corr == vola).

- **No Versioning Strategy**: pyproject.toml shows `version = "0.2.2"` but no CHANGELOG.md or migration guide. Breaking changes between 0.1 → 0.2 are undocumented.

- **Marimo Notebooks Without Validation**: README mentions Marimo notebooks (`make marimo`) but the `book/marimo/` directory structure and validation artifacts (`results/`) are gitignored. No CI validation of notebook correctness.

- **Linear System Normalization**: The position solving normalizes by `inv_a_norm(expected_mu, matrix)` (line 354 in `optimizer.py`) to achieve scale-invariance, but this denominator can be zero or near-zero. Current guard (`<= denom_tol`) zeros positions but doesn't log or warn, making diagnosis difficult.

- **Dependencies on Private Packages**: CI workflows reference `secrets.UV_EXTRA_INDEX_URL` and `GH_PAT` for "private packages" (`.github/workflows/rhiza_ci.yml`). No public documentation on what these are or how contributors can access them.

### Score

**7.5 / 10**

**Justification**:  
Basanos is a solid, well-crafted library with excellent engineering practices, clean code, and strong documentation. It would score 8+ if:
1. Numerical stability concerns were addressed (Cholesky decomposition, condition number checks)
2. Performance characteristics were documented with benchmarks
3. Input validation were more comprehensive (property-based tests, domain checks)
4. The scope were broader (multiple optimization approaches or extensibility for user-defined optimizers)

The library is production-ready for its narrow use case (correlation-aware optimization) but requires careful validation and domain expertise to use correctly. Not suitable as a general-purpose portfolio optimization library. The mathematical implementation is sophisticated and the code quality is high, but the lack of benchmarks, limited error handling, and undocumented numerical assumptions prevent a higher score.

---

## 2026-03-16 — Follow-Up Analysis Entry

### Summary

Basanos has undergone recent improvements addressing several previously identified weaknesses. Recent PRs (#43, #46, #48) fixed hardcoded assumptions documentation, date column dependency issues, and cache pollution. The codebase now features ~2,018 LOC in `src/` and ~2,113 LOC in `tests/` (test-to-source ratio ~1.05), indicating strong test coverage. The library maintains a clean architecture with custom exception types (`NonSquareMatrixError`, `DimensionMismatchError`, `SingularMatrixError`, `InsufficientDataError`) introduced to improve error handling. However, numerical stability concerns remain unaddressed—no Cholesky decomposition or condition number checks are present. The profit variance EMA logic (lines 414-431 in `optimizer.py`) is subtle and lacks dedicated unit tests. Overall, the library shows steady improvement but still lacks benchmarks, property-based tests, and robust numerical safeguards.

### Strengths

- **Improved Exception Handling**: Custom exception hierarchy in `src/basanos/exceptions.py` provides domain-specific context for matrix operations. All exceptions inherit from `BasanosError` for easy catching. Exceptions include detailed messages with actual dimensions/sizes.

- **Enhanced Input Validation**: `BasanosEngine.__post_init__` (lines 241-297) now validates non-positive prices, excessive NaNs (>90% threshold), and monotonic price series. Raises `ValueError` with descriptive messages for each failure mode.

- **Configurable Hardcoded Values**: Recent work exposed previously hardcoded constants (`profit_variance_init`, `profit_variance_decay`, `denom_tol`, `position_scale`) as configurable `BasanosConfig` fields with documented defaults and rationale (lines 176-212).

- **Active Maintenance**: Git history shows 3+ recent PRs merged in March 2026 addressing specific issues. CHANGELOG.md tracks changes properly with semantic versioning (0.2.2 → unreleased). Pre-commit hooks, Renovate, and comprehensive CI workflows ensure code quality.

- **Test Coverage Quality**: 36 test functions in `test_optimizer.py` alone. Tests use fixtures for reproducible scenarios (`optimizer_prices`, `small_prices`, `optimizer_mu`) with seeded RNGs. Total test LOC exceeds source LOC.

- **Documentation Suite**: 12 markdown files in `docs/` covering architecture, testing, security, customization, marimo notebooks, glossary, quick reference, and presentation. README provides clear examples with exact code blocks.

- **Minimal Technical Debt Markers**: Zero TODO/FIXME/XXX/HACK comments in source code. No committed `__pycache__` files (2,110 cache entries exist but are properly gitignored).

- **Rhiza Framework Integration**: Uses modern Python tooling (uv, ruff, marimo) with standardized Makefile targets. GitHub Actions workflows include setup hooks (`.github/workflows/copilot-setup-steps.yml`) and session lifecycle hooks (`.github/hooks/hooks.json`).

### Weaknesses

- **No Property-Based Tests**: Despite availability of Hypothesis and documented infrastructure in `docs/TESTS.md`, no property-based tests exist in `tests/property/`. No tests for edge cases like all-zero mu vectors, perfectly correlated assets, or single-asset portfolios.

- **Benchmark Infrastructure Incomplete**: `.benchmarks/` directory exists but is empty. No pytest-benchmark tests found in test suite. No performance metrics documented for scalability (asset count, time series length, memory usage).

- **Profit Variance Logic Untested**: Critical adaptive position sizing logic (lines 414-453 in `optimizer.py`) uses yesterday's position for today's profit calculation with lagged volatility. No dedicated unit tests validate this complex EMA update mechanism.

- **No Numerical Stability Safeguards**: Despite using `np.linalg.solve` on shrunk correlation matrices, no positive-definiteness checks exist. No Cholesky decomposition, no condition number monitoring. `SingularMatrixError` catches `LinAlgError` post-facto but doesn't prevent ill-conditioned systems.

- **Limited Visualization Options**: Hard dependency on Plotly for all plots. No option for matplotlib, seaborn, or static publication-quality exports. `Plots` class in `src/basanos/analytics/_plots.py` is Plotly-only.

- **Sparse Docstring Coverage in Tests**: Test files lack module-level docstrings and many test functions have minimal documentation. Fixtures are well-documented but test intent isn't always clear without reading implementation.

- **Date Column Assumption Persists**: While PR #46 addressed some date column issues, methods like `Portfolio.truncate` and `Portfolio.monthly` still assume a 'date' column exists. Integer-indexed DataFrames would fail silently or unexpectedly.

- **No Backtesting Utilities**: Portfolio analytics are comprehensive but lack transaction cost modeling, slippage, rebalancing frequency constraints, or out-of-sample validation helpers. Users must build backtesting infrastructure themselves.

### Risks / Technical Debt

- **EMA Correlation Numerical Accuracy**: `_ewm_corr_numpy` (lines 25-117) implements complex IIR filtering via `scipy.signal.lfilter` to match pandas EWM behavior. While documented as "within floating-point rounding error," no explicit numerical stability tests validate this claim across diverse data distributions (sparse data, high volatility, near-singular cases).

- **Matrix Shrinkage Alone Insufficient**: Shrinkage towards identity (`shrink2id`) improves conditioning but doesn't guarantee positive-definiteness when correlation estimates are poor. At `shrink=0.5` (common default), the matrix is still 50% empirical correlation, which could be indefinite with insufficient data.

- **Error Context Loss**: While custom exceptions now exist, many validation checks in `BasanosEngine.__post_init__` still raise generic `ValueError` without using custom exception types. Line 252: `raise ValueError` for missing date column provides no structured context.

- **Undocumented Performance Cliffs**: No documentation on when the optimizer becomes impractical (e.g., >1000 assets, >10 years daily data). Linear algebra operations scale as O(N²T) for correlation computation and O(N³) per solve. Memory usage is undocumented.

- **CI Secrets Dependency**: `.github/workflows/rhiza_ci.yml` references `secrets.UV_EXTRA_INDEX_URL` and `secrets.GH_PAT` for "private packages" (lines 39-43). No public documentation explains these dependencies, making external contributions difficult.

- **Marimo Notebooks Unvalidated**: `docs/MARIMO.md` mentions notebooks in `book/marimo/`, but they're not validated in CI. No tests ensure notebook cells execute correctly or produce expected outputs. Notebooks could be stale or broken.

- **Incomplete Type Hints**: While `py.typed` marker was recently added (CHANGELOG unreleased), actual type hint coverage is unknown. No mypy or pyright runs visible in CI workflows. Type safety is advertised but not systematically enforced.

- **Zero-Division Guards Insufficient**: Line 448 checks `denom <= self.cfg.denom_tol` and `np.allclose(expected_mu, 0.0)` to avoid division by zero, but these checks happen after computing `inv_a_norm`, which can return `np.nan`. The conditional doesn't explicitly handle `denom is None` (which `inv_a_norm` doesn't return—it returns `float(np.nan)`).

### Score

**7.0 / 10**

**Justification**:  
Score decreased by 0.5 points from previous entry (7.5 → 7.0) despite recent improvements. While exception handling and input validation improved, the absence of property-based tests, empty benchmark infrastructure, and persistent numerical stability concerns outweigh these gains. The library demonstrates strong engineering practices and active maintenance, but critical gaps remain:

1. **No property-based tests**: Despite documented infrastructure, no actual tests validate edge cases systematically.
2. **Benchmarks missing**: Infrastructure exists (`.benchmarks/` directory) but contains no data or tests.
3. **Numerical stability unaddressed**: No Cholesky decomposition, condition number checks, or positive-definiteness validation before solving linear systems.
4. **Profit variance logic untested**: Complex risk-scaling mechanism (lines 414-453) lacks dedicated unit tests despite being critical to position sizing.

The library remains production-ready for narrow use cases with expert oversight but is not suitable for general-purpose deployment without:
- Comprehensive property-based tests covering edge cases (singular matrices, zero signals, extreme correlations)
- Published performance benchmarks and scalability limits
- Numerical stability safeguards (Cholesky decomposition, condition number monitoring)
- Dedicated tests for profit variance EMA logic

Recent improvements (custom exceptions, enhanced validation, configurable parameters) demonstrate commitment to quality but don't address fundamental numerical robustness concerns. The test-to-source ratio of 1.05 is excellent, but test quality matters more than quantity—absence of property-based tests and benchmark data is a significant gap for a quantitative finance library.
