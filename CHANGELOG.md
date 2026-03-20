# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `[project.urls]` added to `pyproject.toml` (Homepage, Repository, Issues).
- `Idea` section added to README explaining the core concept and C=I corner case.

### Changed

- `BasanosConfig` now raises an informative `TypeError` (rather than a generic
  Pydantic `extra_forbidden` error) when the pre-v0.4 flat kwargs
  `covariance_mode`, `n_factors`, or `window` are passed directly to the
  constructor.  The error message includes a copy-pasteable migration recipe.

### Migration

#### `BasanosConfig` covariance API (v0.3 → v0.4)

The flat kwargs `covariance_mode`, `n_factors`, and `window` were replaced by
a nested discriminated union field `covariance_config`.

**Before (v0.3 and earlier):**

```python
from basanos.math import BasanosConfig

cfg = BasanosConfig(
    vola=16, corr=32, clip=3.5, shrink=0.5, aum=1e6,
    covariance_mode="sliding_window",
    window=30,
    n_factors=2,
)
```

**After (v0.4+):**

```python
from basanos.math import BasanosConfig, SlidingWindowConfig

cfg = BasanosConfig(
    vola=16, corr=32, clip=3.5, shrink=0.5, aum=1e6,
    covariance_config=SlidingWindowConfig(window=30, n_factors=2),
)
```

For the default EWMA-shrink mode no `covariance_config` argument is needed:

```python
cfg = BasanosConfig(vola=16, corr=32, clip=3.5, shrink=0.5, aum=1e6)
```

## [0.2.3] - 2026-03-16

### Added

- `Stats.sharpe()` now returns `float('nan')` instead of raising `ZeroDivisionError`
  when the series has zero volatility (constant returns).
- `py.typed` marker added to the package so that PEP 561 type-checking discovery
  works correctly for downstream consumers.
- `cor_tensor` property on `BasanosEngine` exposing stacked correlation matrices;
  flat-file round-trip (`.npy`) added and tested.
- Pure NumPy EWM correlation via `lfilter`; pandas and pyarrow removed as runtime
  dependencies (retained as optional dev dependencies for benchmarking).
- `BasanosConfig` dataclass consolidating optimizer constants (replaces hardcoded values).
- Data quality validation in `BasanosEngine.__post_init__`.
- `MissingDateColumnError` and `IntegerIndexBoundError` custom exception classes for
  explicit date-column error handling.
- Cholesky-based solving and condition number monitoring for numerical stability.
- Explicit `is_positive_definite` check prior to Cholesky decomposition.
- Logging warning emitted when the normalisation denominator is degenerate.
- Property-based tests using `hypothesis`.
- `ty` added as a dev dependency; type-checking CI workflow added.
- Marimo demo notebook.
- Shrinkage methodology guide notebook.
- Benchmark baseline results published.
- Optimizer performance cliffs documented (complexity, memory, practical limits).

### Fixed

- Zero-division guards: pre-check `expected_mu`, remove dead `denom is None` branch,
  suppress spurious NaN divide warnings.
- Exception context loss: all re-raises now use `raise … from` to preserve tracebacks.
- Date column dependency resolved in `Portfolio` methods.
- Cache pollution: pre-commit hook prevents committing `__pycache__` / `.pyc` files.
- Shrinkage guide notebook cell failures resolved.
- SECURITY.md supported-versions table updated to reflect actual releases.

### Changed

- 100% docstring coverage across all source modules; doctests added throughout.
- Type hint coverage improved throughout the codebase.

## [0.2.2] - 2026-03-10

### Added

- `Stats.summary()` convenience method returning a labelled performance summary.

## [0.2.1] - 2026-03-10

No functional changes; version alignment after initial packaging.

## [0.2.0] - 2026-03-10

### Changed

- Professional README with usage examples added.

## [0.1.0] - 2026-03-02

Initial release.
