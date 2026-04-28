# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking:** All IC functions (`ic`, `rank_ic`, `ic_mean`, `ic_std`, `icir`,
  `rank_ic_mean`, `rank_ic_std`) converted from properties to methods accepting
  a horizon parameter `h: int = 1`. Existing code using `engine.ic` must be
  updated to `engine.ic()`. Use `h > 1` to evaluate signal quality against
  multi-period forward returns (e.g. `engine.ic(h=5)` for 5-day IC).

## [0.6.0] - 2026-03-27

### Added

- Per-class navigable API reference via MkDocs + mkdocstrings.
- CI matrix expanded to Ubuntu + macOS.
- PyPI badge added to README.
- GitHub Discussion templates (help-wanted and ideas).
- Link-check workflow (`link-check.yml`).
- GitHub release changelog config (`.github/release.yml`).
- Smoke test for paper §6 code listing.

### Changed

- Docs migrated from pdoc/minibook to MkDocs.
- `uv sync` added to CONTRIBUTING and linked from README.
- Paper (`basanos.tex`): restructured flow, added latent SVD factors,
  Sherman–Morrison identity, and sliding-window section.
- `actions/checkout` updated to v6; `actions/deploy-pages` updated to v5.

### Fixed

- EWM correlation matrix now enforces exact symmetry.
- Broken coverage badge link fixed.

## [0.5.1] - 2026-03-24

### Added

- `pytest-xdist` for parallel test execution.

### Changed

- `basanos.analytics` subpackage retired; replaced by `jquantstats`.
- Mixin inheritance restored in `BasanosEngine` to fix IDE Go-to-Definition.

### Removed

- Profit-variance EMA feature removed entirely.

## [0.5.0] - 2026-03-21

### Added

- `BasanosStream` class for O(N²) incremental position updates with
  `step()`, `from_warmup()`, `save()`, and `load()` methods.
- `StepResult` frozen dataclass as the output type for `BasanosStream.step()`.
- `_StreamState` mutable dataclass carrying O(N²) IIR state.
- `SlidingWindowConfig` streaming support in `BasanosStream`.
- Format version tag added to `BasanosStream.save()/load()` for forward
  compatibility.
- `max_components` field on `SlidingWindowConfig` to expose and document
  SVD truncation.
- `max_turnover` constraint applied in `BasanosStream.step()`.
- Position-delta cost model and turnover budget constraint in `BasanosEngine`.
- `MatrixBundle` dataclass to future-proof `_compute_position` against
  argument-list growth.
- `SolveStatus` enum replacing magic status strings.
- End-to-end worked example notebook.
- conda-forge recipe.
- Numerical precision and regression tests against analytical solutions.

### Changed

- `EwmaShrink` solve loop vectorised via batched `numpy.linalg.solve`.
- `_iter_solve` refactored: extracted `_compute_position`, delegated to
  `_iter_matrices`; dual-path divergence documented with cross-path
  consistency test.
- `BasanosStream.from_warmup()` decoupled from `BasanosEngine._iter_solve`
  private API.
- `StepResult.status` narrowed from `str` to `Literal` type.

### Fixed

- `fill_nan(0.0)` added to `position_delta_costs` to clear EWMA warmup NaN.
- Sliding-window NaN-padding warmup semantics documented and tested.
- `SlidingWindowConfig` validates `max_components <= n_factors`.

## [0.4.3] - 2026-03-20

### Added

- `BasanosConfig` now raises an informative `TypeError` when pre-v0.4 flat
  kwargs (`covariance_mode`, `n_factors`, `window`) are passed directly;
  error message includes a copy-pasteable migration recipe.
- `CovarianceConfig` type alias exported from `basanos.math`.
- `position_status` property on `BasanosEngine` distinguishing
  warmup / zero_signal / degenerate / valid rows.
- Diagnostics Marimo notebook for `position_status`, `condition_number`, and
  `solver_residual`.
- Jinja2 HTML templates for report generation.
- License and repository metadata added to `pyproject.toml`.
- Weekly dep-compat CI job to detect API-breaking upstream releases.
- Repository analysis journal (`REPOSITORY_ANALYSIS.md`).

### Changed

- `BasanosEngine` decomposed from 83 KB God Object into focused sub-modules.
- `Portfolio` refactored to composition over inheritance from `PortfolioData`.
- Exception types consolidated from 25 to 18.
- Mixin contracts replaced with Protocol-based typing.
- `EwmaShrink` and `_engine_solve` split into separate sub-modules.
- `make` example modules removed; active build system documented.
- Upper-bound dependency pins relaxed.

### Fixed

- Malformed Issues URL in `pyproject.toml` corrected.
- `polars` upper bound removed to eliminate manual version-bump burden.
- Protocol annotation on `_reject_legacy_flat_kwargs` narrowed.
- Dead `isinstance` guard removed.

## [0.4.2] - 2026-03-19

### Added

- `FactorModel` frozen dataclass.
- `FactorModel.solve()` method using Cholesky-based Woodbury inner system.
- Factor-model guide Marimo notebook.
- 100% test coverage (added `SingularMatrixError` path coverage).

### Fixed

- `pragma: no cover` assertions reviewed; dead vol helper removed from
  `portfolio.py`; domain exceptions given descriptive messages.

## [0.4.1] - 2026-03-19

Minor sync release; no functional changes.

## [0.4.0] - 2026-03-19

### Added

- Sliding-window factor model (`SlidingWindowConfig`) as an alternative to
  EWMA/shrinkage covariance (paper Section 4.4).
- `BasanosConfig` discriminated union field `covariance_config` replacing
  the flat kwargs `covariance_mode`, `n_factors`, and `window`.
- `ConfigReport` for `BasanosConfig` parameter analysis and lambda sweep.
- Trading cost impact analysis added to `Portfolio`, `Plots`, and `Report`.
- Companion LaTeX paper (`paper/basanos.tex` + `paper/basanos.bib`).
- CI workflow to compile LaTeX and publish PDF artifact to `paper` branch.
- Structured JSON logging via `JSONFormatter`.
- Integration/smoke tests for the full `BasanosEngine` pipeline.
- Marimo notebook catalog (`docs/MARIMO.md`).
- `_MIN_CORR_DENOM` and `_MAX_NAN_FRACTION` consolidated into `BasanosConfig`.
- Coverage badge via CI and `genbadge`.
- Diagnostics properties on `BasanosEngine` (condition number, solver
  residual, signal utilisation, etc.).

### Changed

- `BasanosConfig` now accepts `covariance_config` (discriminated union) in
  place of the former flat kwargs.  See migration guide below.
- Paper updated: added factor risk models, latent SVD factors,
  Sherman–Morrison identity, and sliding-window section.
- `_validate_inputs` extracted from `__post_init__` as a standalone helper.
- `_data` declared as `ClassVar` on `Portfolio` to remove `type: ignore`.
- `cor` key type narrowed from `dict[object, …]` to `dict[datetime.date, …]`.

### Fixed

- `SingularMatrixError` caught in `solver_residual` and `signal_utilisation`.
- Descriptive error messages added to `Portfolio.__post_init__()` validation.
- Coverage badge CI fixed (now uses `coverage.xml` and `genbadge[coverage]`).
- `analyse-repo` workflow failures resolved.

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

## [0.3.0] - 2026-03-17

### Added

- HTML report generation: `Report` class with monthly heatmap.
- Turnover measures: `Portfolio.turnover`, `turnover_weekly`,
  `turnover_summary`, and shrinkage-guide sweep.
- Additional `Stats` metrics: rolling Sharpe, rolling volatility, annual
  breakdown, temporal performance metrics (`win_rate`, `profit_factor`,
  `payoff_ratio`, `monthly_win_rate`, `worst_n_periods`, up/down capture).
- Maximum drawdown measures.
- IC, Rank IC, and ICIR metrics on `BasanosEngine`.
- Six diagnostic properties on `BasanosEngine`.
- Property-based tests using `hypothesis`.
- 100% test coverage.

### Fixed

- Date column preserved when scaling cash positions in portfolio.
- Polars `Series` stat return types narrowed to satisfy type-checker.

## [0.2.4] - 2026-03-17

### Added

- `[project.urls]` added to `pyproject.toml` (Homepage, Repository, Issues).
- `Idea` section added to README explaining the core concept and C=I corner
  case.

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
