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

---

## 2026-03-19 — Analysis Entry #3

### Summary

Three PRs merged since entry #2 close two of the three issues filed in that session (#222, #223) and go significantly further than asked on the config pattern (#229). The headline change is a structural refactor of `BasanosConfig`: the `covariance_mode / window / n_factors` flat fields are replaced by a Pydantic discriminated union `covariance_config: EwmaShrinkConfig | SlidingWindowConfig`. This removes the cross-field `model_validator`, eliminates the `int | None` tech debt, and makes required sliding-window parameters structurally required at the type level. Issue #224 (warm-up gap documentation) remains open.

### Strengths

- **Discriminated union pattern is architecturally correct** (`optimizer.py:325`): `CovarianceConfig = Annotated[EwmaShrinkConfig | SlidingWindowConfig, Field(discriminator="covariance_mode")]` is the canonical Pydantic v2 solution. Pydantic will raise a clear validation error if a `SlidingWindowConfig` is missing `window` or `n_factors` — no custom validator needed. The `model_validator` is cleanly removed.
- **Backward-compatible properties preserved**: `BasanosConfig.covariance_mode`, `.window`, `.n_factors` remain as computed `@property` accessors (`optimizer.py:589–602`), so call sites outside `optimizer.py` (tests, user code) require no changes.
- **`isinstance` dispatch replaces enum comparison**: `if isinstance(self.cfg.covariance_config, EwmaShrinkConfig)` is both type-safe (mypy/ty infer the narrowed type) and forward-compatible (a third mode requires only a new sub-config class, not a new enum value + validator branch).
- **`cast` use now cleaner**: `sw_config = cast(SlidingWindowConfig, self.cfg.covariance_config)` after the `isinstance` check is technically redundant but serves as a type-narrowing hint for static analysis. The previous double-cast pattern (`cast(int, self.cfg.window)`) is gone.
- **Sub-configs exported from public API**: `EwmaShrinkConfig` and `SlidingWindowConfig` are exported from `basanos.math.__init__`. Users constructing configs programmatically no longer need to import from the private `optimizer` module.
- **Except clauses narrowed with logging** (PR #227): All three broad `except Exception` blocks in `_iter_matrices` and `cash_position` now catch `(np.linalg.LinAlgError, ValueError)` and log at `WARNING` / `DEBUG` level. Production fallbacks are now observable without code changes.
- **Benchmark baseline refreshed and extended** (PR #226): `BENCHMARKS.md` now shows v0.4.2 results (replacing stale v0.3.0), covers six sliding-window cases (`sw_252_5_60_3`, `sw_252_20_60_5`, `sw_1260_5_60_3`, etc.), and the CI regression gate covers the new path.

### Weaknesses

- **Issue #224 still open — warm-up gap undocumented**: `SlidingWindowConfig.window` field description (`optimizer.py:289–293`) still reads only *"Rule of thumb: W >= 2 * n_assets"*. The W-1 row warm-up period producing zero positions is not mentioned. The note at `_iter_matrices` docstring line 833 ("before the warm-up period has elapsed") exists but is in an internal method, invisible to users reading `BasanosConfig`.
- **`EwmaShrinkConfig` is an empty model**: The class body is `covariance_mode: Literal[...] = ...` only. This is correct for now but means any future ewma-specific parameters (e.g., a `corr_override`) would require a potentially breaking change to the field name convention. A comment or docstring note about intentional minimalism would clarify intent.
- **`CovarianceConfig` type alias not exported**: `CovarianceConfig` (the `Annotated` discriminated union alias) is used as the type of `BasanosConfig.covariance_config` but is not exported from `basanos.math.__init__`. Type-annotated user code that wants to declare a variable of this union type must import from the private `optimizer` module.

### Risks / Technical Debt

- **Three open copilot branches unmerged at time of writing**: `copilot/document-warm-up-gap-in-basanosconfig`, `copilot/fix-paper-pdf-issue`, `copilot/narrow-except-exception-errors` (the last may be superseded by the merged PR #227). Branch accumulation without merge creates review debt.
- **`SlidingWindowConfig` constructor in `sharpe_at_pca_components`** (`optimizer.py:1378`): The `model_copy(update={"covariance_config": SlidingWindowConfig(...)})` pattern works but is fragile — any frozen field added to `BasanosConfig` that has no default would break the copy silently. Prefer a dedicated factory or constructor kwarg forwarding.
- **Benchmark commit hash is stale**: `BENCHMARKS.md` records `Commit: 9aa4491` as the baseline, which is two merges behind the code that actually generated the results. Minor, but reduces traceability for regression investigations.

### Score

**9/10** — Score unchanged; structural config quality raised, one open issue remains

**Rationale**:
The discriminated union refactor is the right long-term design and was delivered quickly. The codebase is now in a cleaner state than it was after the initial sliding-window merge. The one remaining open item (#224) is low-severity. No regressions introduced.

---

## 2026-03-20 — Analysis Entry #4

### Summary

Seven PRs merged since entry #3 systematically close six of the eight weaknesses and risks catalogued in that session. The most significant addition is `position_status` (PR #236): a new `BasanosEngine` property that labels every `cash_position` row as `warmup`, `zero_signal`, `degenerate`, or `valid`, directly resolving the long-standing "indistinguishable output rows" weakness. The Woodbury `np.linalg.inv` issue is fixed, the dead-code `pragma: no cover` branches are removed from `cash_position`, the singular-matrix edge case is pinned as a deterministic test, and the breaking API change is softened with an explicit `TypeError` carrying a migration guide. Coverage dips slightly to 99.27% (570 tests, 1,640 statements). One prior weakness (#224, warm-up gap undocumented at field level) remains open.

### Strengths

- **`position_status` closes the observability gap** (`optimizer.py:1124`): The four-value status enum (`warmup` / `zero_signal` / `degenerate` / `valid`) gives downstream consumers (backtests, risk monitors) a first-class reason code for every row without requiring re-inspection of `mu` or engine config. The docstring is thorough, the return type (`pl.DataFrame` with `date` + `status`) is consistent with the rest of the engine API, and 88 lines of new tests cover all four states.
- **Woodbury path is now numerically consistent** (`_factor_model.py`): PR #235 replaces `np.linalg.inv(self.factor_covariance) + ...` with `_cholesky_solve`, aligning the factor model with the Cholesky-first discipline applied everywhere else in `_linalg.py`. The fix is 7 lines; the quality improvement is structural.
- **Dead-code branches removed from `cash_position`** (PR #239): The two `k_eff < 1` blocks annotated `# pragma: no cover` in `_iter_matrices` and the sliding-window `cash_position` loop are gone. Coverage reporting is now honest.
- **Singular-matrix edge case promoted to deterministic test** (PR #243): The Hypothesis-discovered `SingularMatrixError` falsifying example is now a standalone test in `test_optimizer.py` (+91 lines). It no longer depends on Hypothesis replay state and will reliably catch regressions on fresh CI runners.
- **Migration ergonomics addressed** (PR #246): `BasanosConfig._reject_legacy_flat_kwargs` catches calls using the pre-v0.4 flat kwargs (`covariance_mode`, `window`, `n_factors`) and raises a `TypeError` with an explicit before/after migration example. This is significantly better than the generic Pydantic `extra_forbidden` error users would otherwise encounter. The message includes import instructions and covers both modes.
- **All three prior export/doc weaknesses closed**: `CovarianceConfig` is now exported from `basanos.math.__init__` (PR #254); `EwmaShrinkConfig` has a docstring explicitly stating intentional minimalism (PR #255); `BENCHMARKS.md` environment table updated to v0.4.2 (PR #247).

### Weaknesses

- **`position_status` duplicates `cash_position` loop logic** (`optimizer.py:1124–1221`): The ~95-line implementation re-implements the core optimizer loop — mask computation, ewma/sliding dispatch, `FactorModel.from_returns`, `inv_a_norm` / `fm.solve`, denom comparison — producing status codes instead of positions. Any future change to the core loop (new edge case, new covariance mode, denom logic change) must be applied in two places. This is the most significant new technical debt introduced in this sprint. The correct long-term fix is to produce status codes as a side-channel of `cash_position` rather than recomputing from scratch.
- **`k_eff < 1` dead-code re-introduced in `position_status`** (`optimizer.py:1198`): PR #239 removed this pattern from `cash_position` but `position_status` reintroduces the identical `if k_eff < 1: # pragma: no cover` branch. The fix was incomplete — the same unreachable branch now exists again one method later.
- **Warm-up gap still undocumented at field level** (`optimizer.py:296–303`): `SlidingWindowConfig.window`'s `Field(description=...)` still reads only "Rule of thumb: W >= 2 * n_assets". The W-1 row warm-up period is documented in the `__post_init__` warning (line 782), in the `_iter_matrices` docstring (line 895), and now in `position_status`'s docstring (line 1129) — but not where a user first reads it: the field description on the config class. Issue #224 remains open.
- **Coverage regression**: 99.27% (12 uncovered statements) vs. 100% in the prior two entries. The regression is attributable to the re-introduced `pragma: no cover` branch in `position_status`. Not critical given the 90% minimum gate, but the 100% invariant was previously treated as a hard bar.

### Risks / Technical Debt

- **`position_status` consistency risk**: Because `position_status` replicates the optimizer loop rather than sharing it, the two can silently diverge. If `cash_position` is updated to handle a new numerical case and `position_status` is not, the status property will misclassify rows. There are no cross-property consistency tests (e.g., asserting that rows labelled `valid` have non-zero positions, or `warmup` rows have NaN positions) that would catch this drift.
- **No streaming / incremental API**: Unchanged from prior entries. `cash_position` and `position_status` both materialise full `(T, N)` results. Live systems appending one row at a time must re-run the full computation.
- **Polars upper bound maintenance burden**: Unchanged. `polars>=1.37.1,<2` requires manual bumps; no compatibility shim.

### Score

**9/10** — Unchanged from prior entries

**Rationale**: The sprint closed six of eight prior items with focused, minimal PRs. The Woodbury fix and singular-matrix promotion are correctness improvements; the migration `TypeError` and `position_status` are meaningful user-experience improvements. The score does not increase because `position_status` introduces a new and significant DRY violation that re-creates the core loop outside the optimizer's natural abstraction boundary — a design debt that will compound as the covariance mode set grows. The `k_eff < 1` dead-code re-appearance and coverage regression are minor but indicate the fix in PR #239 was not applied holistically.

---

## 2026-03-20 — Analysis Entry #5

### Summary

Five PRs merged since entry #4 close every weakness and risk filed in that session. The headline change is the `_iter_solve` generator (PR #263): a single authoritative private method that yields `(i, t, mask, pos_or_none, status)` per timestamp, collapsing the two near-identical optimizer loops that previously existed in `cash_position` and `position_status` into one. `cash_position` and `position_status` are now thin consumers of `_iter_solve`. Additionally, `BasanosConfig.replace()` (PR #264) replaces the fragile `model_copy(update={...})` pattern with an explicitly typed constructor-forwarding method, and the warm-up gap documentation (issue #224) is finally closed at the field level (PR #256). Coverage recovers to 99.81% with 580 tests. Zero open issues remain from the two prior sprints.

### Strengths

- **`_iter_solve` closes the DRY violation cleanly** (`optimizer.py:1025`): The generator is 70+ lines and fully documented with a `Yields` section describing all five tuple elements. `cash_position` (`optimizer.py:1142`) and `position_status` (`optimizer.py:1214`) are now ~20-line wrappers that consume it — no logic duplication. This is structurally correct: the loop now has a single implementation that cannot diverge between the two public properties. The approach also makes it straightforward to add a third consumer (e.g., a streaming API) without touching the core logic.
- **`BasanosConfig.replace()` is a typed, forward-compatible pattern** (`optimizer.py:620`): The new method enumerates all 12 `BasanosConfig` fields explicitly and passes them to the constructor. Unlike `model_copy(update={...})`, any new required field added to `BasanosConfig` will immediately surface as a missing-argument lint error at every `replace()` call site. The `sharpe_at_pca_components` use case that motivated the risk note in entry #3 is now safe. 36 lines of tests added.
- **Cross-property consistency tests added** (`test_optimizer.py:+157 lines`, PR #261): Tests assert that rows labelled `valid` have ≥1 non-zero position, `warmup` rows are NaN, `zero_signal` / `degenerate` rows are zero. These tests are parametrised over both covariance modes. The skip-rather-than-fail design (4 skipped when a status code doesn't appear in the fixture) is pragmatic — avoids false failures while still exercising every path that the fixture can produce.
- **Warm-up gap documented at field level** (PR #256, `optimizer.py:296–303`): `SlidingWindowConfig.window`'s `Field(description=...)` now explicitly states the W-1 row warm-up semantics. Issue #224, open since the initial sliding-window merge, is finally closed.
- **Dead-code branch removed from `position_status`** (PR #262): The `if k_eff < 1: # pragma: no cover` re-introduced in entry #4 is gone. The fix is now complete and consistent across the entire codebase.
- **Coverage recovery**: 99.81% (580 tests, up from 570) vs. 99.27% in entry #4. The regression is almost fully reversed.

### Weaknesses

- **One `pragma: no cover` remains** (`optimizer.py:1091`): The `except SingularMatrixError` block inside `_iter_solve`'s EwmaShrink path — reached only when `solve(matrix, expected_mu)` fails after `inv_a_norm` already succeeded with a finite denominator — is marked uncovered. This is a genuinely rare edge (matrix passes the norm check but fails the solve), so the annotation is defensible, but it prevents 100% coverage recovery.
- **Skipped consistency tests depend on fixture coverage**: The 4 skipped tests (`warmup`, `zero_signal`, `degenerate` status checks) skip at runtime when the fixture doesn't produce that code. If a future refactor changes which status codes appear in the standard fixtures, the consistency guarantee silently loses coverage rather than failing loudly. Dedicated fixtures that force each status code would be more robust.
- **`_iter_solve` docstring mentions `pos_or_none` as `None` for warmup/no-price rows**, but `cash_position` uses `NaN` for warmup and `0` for no-price. The distinction is handled correctly in `cash_position`'s consumer logic, but the tuple contract mixes sentinel semantics (`None` vs zero array) that a future consumer must understand to implement correctly.

### Risks / Technical Debt

- **No streaming / incremental API**: Unchanged from prior entries. `_iter_solve` materialises all timestamps in one pass; live systems still must re-run the full computation per new row.
- **Polars upper bound maintenance burden**: Unchanged. `polars>=1.37.1,<2` requires manual bumps.

### Score

**9/10** — Unchanged; technical debt from entry #4 fully cleared, no new issues introduced

**Rationale**: All four items filed in entry #4 are resolved, and the implementation quality is genuinely higher: `_iter_solve` is the right abstraction, `BasanosConfig.replace()` is the right pattern, and the consistency tests provide a real regression guard. The score remains 9/10 rather than increasing because the two structural limits — no incremental API and the Polars ceiling — are unchanged, and the single remaining `pragma: no cover` indicates one untested failure mode in the core loop. The codebase is in its cleanest state to date.

---

## 2026-03-20 — Analysis Entry #6

### Summary

Four PRs merged since entry #5 address all three weaknesses and both risks filed in that session. The Polars upper bound is removed (PR #268), the `_iter_solve` docstring has a precise RST contract table for `(status, pos_or_none)` combinations (PR #272), a deterministic test covers the previously-uncovered `SingularMatrixError` branch (PR #270), and dedicated fixtures for all four `position_status` codes are added (PR #269). Coverage holds at 99.81% (584 tests); 3 statements remain uncovered and 4 tests still skip. The skip issue is partially resolved — new dedicated tests exist but the old skip-based tests were not removed.

### Strengths

- **Polars upper bound removed** (`pyproject.toml`): `polars>=1.37.1,<2` → `polars>=1.37.1`. This eliminates the manual-bump maintenance burden noted since the initial analysis. The change is one character; the benefit is permanent removal of a recurring chore.
- **`_iter_solve` tuple contract fully specified** (`optimizer.py:1025`): The docstring now contains an RST list-table mapping every `(status, pos_or_none)` combination to its downstream effect in `cash_position`. `None` is now exclusive to `'warmup'`; all other statuses yield an `np.ndarray` (possibly zero-length). Future consumers can branch on `pos_or_none is None` without inspecting `status`. This is the right contract and it is clearly documented.
- **`SingularMatrixError` branch covered** (`test_optimizer_edge_cases.py:+45 lines`, PR #270): The previously `# pragma: no cover` annotated branch is now exercised by a deterministic test that mocks `_cholesky_solve` to raise after `inv_a_norm` succeeds. The annotation is removed; the branch is verified correct.
- **Dedicated `position_status` fixtures added** (PR #269, `test_optimizer.py:+158 lines`): Fixtures are constructed to guarantee each status code appears at least once — a `SlidingWindowConfig` engine with insufficient history for warmup, a zero-`mu` engine for zero-signal, a NaN-price engine for degenerate, and a valid engine. These tests no longer depend on the standard fixture incidentally producing the right rows.

### Weaknesses

- **Old skip-based consistency tests not removed** (`test_optimizer.py:2635,2650,2665`): PR #269 added dedicated fixtures alongside the skip-based tests but did not remove the originals. The result is that 4 tests still skip on every run, and the same consistency property is now tested twice — once robustly (new dedicated fixtures) and once weakly (old skip guards). This is test-suite noise, not a correctness risk, but it contradicts the stated intent of issue #266.
- **Sliding-window no-price degenerate path uncovered** (`optimizer.py:1129–1130`): The `not mask.any()` → `yield ..., np.zeros(0), "degenerate"` branch in the sliding-window arm of `_iter_solve` has no test. The analogous EwmaShrink branch is covered. A fixture with at least one all-NaN price row under `SlidingWindowConfig` would close this gap.
- **`_reject_legacy_flat_kwargs` non-dict path uncovered** (`optimizer.py:601`): The `if not isinstance(data, dict): return data` early-exit is never exercised. Pydantic passes a dict for normal construction; the non-dict path is reached only during model deserialization from an existing instance. Minor, but contributes to the 3 remaining uncovered statements.

### Risks / Technical Debt

- **No streaming / incremental API**: Unchanged.

### Score

**9/10** — Unchanged

**Rationale**: The sprint closes all filed items. The Polars ceiling removal is a long-outstanding dependency hygiene win. The `_iter_solve` contract table is exemplary documentation. The score does not increase because the two remaining coverage gaps (`optimizer.py:1129-1130` and `:601`) are small but real, the old skip tests were not cleaned up, and the structural streaming gap remains. The codebase continues to demonstrate strong engineering discipline; these are polish items rather than material concerns.

---

## 2026-03-20 — Analysis Entry #8

### Summary

Six commits across four PRs since entry #7 close all five weaknesses and both risks filed in that session, then continue the God Object decomposition with a second split pass: `optimizer.py` sheds `_ewm_corr_numpy` into `_ewm_corr.py` (126 lines) and `_SolveMixin` into `_engine_solve.py` (241 lines), reducing the main file from ~1,033 to 716 lines. A `typing.Protocol` (`_EngineProtocol`) replaces the annotation-only attribute stubs and the `raise NotImplementedError` sentinel in `_DiagnosticsMixin`, making the mixin contract formally verifiable. `problem.md` is deleted (superseded by the compiled paper), version bumped to 0.4.3, and Jinja2 upper bound removed. Test suite reaches 656 tests. The module decomposition is effectively complete; the remaining open items are structural design decisions rather than engineering gaps.

### Strengths

- **`_EngineProtocol` is the correct mixin contract pattern** (`_engine_protocol.py:31`): A `typing.Protocol` with 6 attribute declarations and 2 method stubs fully replaces the annotation-only class variables from entry #7. Type checkers (`ty`, `mypy`) can now verify statically that any class used as a mixin host satisfies the contract. The `self: _EngineProtocol` self-typing in `_SolveMixin`, `_DiagnosticsMixin`, and `_SignalEvaluatorMixin` methods is the accepted pattern for structural Protocol constraints on `self` in Python mixins — not standard `self`, but unambiguous and understood by all major type checkers.
- **`_ewm_corr_numpy` stands alone** (`_ewm_corr.py:14`): The 126-line module is arguably the best-documented unit in the codebase. It states its pandas-parity contract, explains why `ignore_na=False` forces joint-finite masking, documents the IIR structure and all 14 peak arrays with sizes, and includes complexity tables. Separating it enables profiling and testing without loading `BasanosEngine` machinery. The mathematical notation in comments (e.g., `v_x[t,j,i] carries x_j[t]`) eliminates the usual risk of the implementation diverging silently from its rationale.
- **`_SolveMixin` extraction completes the core decomposition** (`_engine_solve.py:28`): The 241-line module is entirely focused on per-timestamp solve logic (`_iter_matrices`, `_iter_solve`). `optimizer.py` no longer contains any of the hot-path computation — it is now a dataclass facade wiring `_DiagnosticsMixin`, `_SignalEvaluatorMixin`, and `_SolveMixin` together, with its own scope limited to input validation, EWM data preparation, and position/portfolio aggregation.
- **`optimizer.py` is 716 lines, down from 1,033** (31% reduction): The decomposition across six modules is now structurally complete. The module-level docstring (`optimizer.py:62–78`) explicitly enumerates all seven sub-modules with their responsibilities, forming a navigable architecture overview.
- **`_reject_legacy_flat_kwargs` dead guard removed cleanly** (`_config.py:412`, PR #293): Annotating `data` as `dict[str, object]` rather than `object` removes the `isinstance` guard at the source — the type annotation itself encodes the invariant that Pydantic always passes a dict, eliminating both the dead code and the coverage annotation.
- **`problem.md` deleted**: The standalone problem statement is superseded by `paper/basanos.tex` and the `make paper` build. Removing it eliminates a divergence risk — there is now a single authoritative problem definition.
- **All three remaining dependency ceilings removed**: Jinja2 `<4` removed in PR #292. Combined with the Polars ceiling removal in entry #6 and the earlier NumPy/SciPy practices, the dependency configuration is now in its cleanest state: only Pydantic (`<3`) and scipy/numpy (`<2`, `<3`) retain upper bounds — both justified by known API stability patterns.

### Weaknesses

- **`cor` key type is `dict[object, np.ndarray]`** (`optimizer.py:299`, `_engine_protocol.py:43`): The `@property` and its Protocol mirror both use `object` as the dict key type. The actual keys are polars date values (the output of `self.prices["date"].to_list()`). `object` admits any key and defeats static analysis of any code that indexes `cor[t]`. Narrowing to `dict[datetime.date, np.ndarray]` or `dict[Any, np.ndarray]` with a comment would be more informative; the former would actually catch key-type mismatches.
- **`PortfolioData`/`Portfolio` IS-A coupling** (`analytics/portfolio.py`): Unchanged from entry #7. `Portfolio(PortfolioData)` means `isinstance(p, PortfolioData)` is true for any `Portfolio` instance. This is likely intentional (the two share all fields) but is not documented as a deliberate design decision. An ADR or a docstring note on the IS-A vs. composition trade-off would signal intent and prevent a future contributor from refactoring toward composition on the assumption the inheritance is accidental.
- **`__post_init__` validation is 60+ lines in the engine facade** (`optimizer.py:199–258`): The six validation checks (date column, shape, column parity, non-positive prices, excessive NaNs, monotonic prices, warm-up gap) could live in a dedicated `_validate_inputs` private method or even a helper module, keeping `__post_init__` to a summary call. This is a minor readability issue at 716 lines total, but worth noting as `optimizer.py` approaches its final form.

### Risks / Technical Debt

- **No streaming / incremental API**: Unchanged from all prior entries. `_iter_solve` and `_iter_matrices` process all `T` timestamps in one pass. Live systems must re-run the full computation per new row. The `_SolveMixin` extraction actually makes this addressable in future: a streaming variant could inherit the same mixin and override `__iter__` behaviour.
- **`self: _EngineProtocol` pattern not documented for contributors**: The mixin methods use an unconventional `self` type annotation. New contributors adding a fourth mixin or extending an existing one may default to annotation-only attributes (reproducing the entry #7 weakness). A brief note in `_engine_protocol.py` or `CONTRIBUTING.md` on the expected pattern would prevent regression.

### Score

**9/10** — Unchanged

**Rationale**: The second decomposition pass finishes what entry #7 started: `optimizer.py` is now a clean facade with no hot-path logic remaining in it. The Protocol-based mixin contract is the correct long-term design. Every weakness and risk from entry #7 is resolved. The score does not increase because the two structural limits — no incremental API and the `PortfolioData`/`Portfolio` inheritance coupling — are unchanged, and the `dict[object, ...]` key type on `cor` is a precision gap in the type contract that the Protocol work otherwise made exemplary. The codebase architecture is in its best state to date.

---

## 2026-03-20 — Analysis Entry #7

### Summary

Seven commits and four PRs merged since entry #6 close all three weaknesses and resolve the last coverage gaps filed in that session, then immediately advance the codebase with four structural improvements: `Portfolio` is split into `PortfolioData` + `Portfolio` analytics facade (PR #280), `exceptions.py` is consolidated from 25 to 18 concrete types (PR #284), the 83 KB `optimizer.py` God Object is decomposed into four focused sub-modules (PR #287), and inline HTML string generation is migrated to Jinja2 templates (PR #288). Test suite grows to 612 tests across 1,616 statements at 99.88% coverage. The entry also records that the two remaining uncovered statements are artefacts of the God Object refactor's type-checker fix rather than genuinely untestable branches.

### Strengths

- **God Object decomposition is architecturally sound** (PR #287): `optimizer.py` drops from ~5,000 to 1,033 lines by extracting `BasanosConfig` and covariance sub-configs into `_config.py` (557 lines), matrix diagnostics into `_engine_diagnostics.py` (213 lines), and IC signal evaluation into `_engine_ic.py` (191 lines). Each module has a clear single responsibility. Public API is unchanged — `BasanosEngine` still inherits both mixins and re-exports everything from `basanos.math`. The decomposition is exactly right: `optimizer.py` now contains only the core solve loop, `_iter_matrices`, and position/leverage properties.
- **`PortfolioData` / `Portfolio` split follows the same mixin discipline** (PR #280): `PortfolioData` (frozen dataclass, 332 lines) holds all fields, validation, and factory methods (`from_risk_position`, `from_cash_position`). `Portfolio(PortfolioData)` adds only analytics, plots, and report generation. `PortfolioData` is exported from the public API. 27 new tests in `test_portfolio_data.py` cover the new class in isolation. The pattern is consistent with `_DiagnosticsMixin` / `_SignalEvaluatorMixin` extracted the same day — a coherent session-wide refactoring philosophy.
- **Exception consolidation reduces surface area** (PR #284): `NullsAfterCleaningError` and `NonFiniteAfterCleaningError` (and 5 other types) are merged into `CleaningInvariantError` and a smaller set of cohesive types. Exception hierarchies are a public API; 18 types are markedly easier to document, import, and `except`-catch than 25. The consolidation is non-breaking for any code that only caught the base `BasanosError`.
- **Jinja2 templates separate presentation from logic** (PR #288): `_report.py` and `_config_report.py` previously built HTML via inline string concatenation. Both now render via `jinja2.Environment(FileSystemLoader(...))` with templates in `src/basanos/templates/`. The `_base.html` parent template handles head/CSS; `portfolio_report.html` and `config_report.html` extend it. This is the correct design for report generation code that will evolve: template changes require no Python edits and are diffable as HTML.
- **All entry #6 coverage gaps closed before the refactor commits**: Commits `36b1373`, `f539e56`, `4aaec94`, and `8706d4b` closed the three remaining uncovered paths and removed skip-based consistency tests prior to the structural changes. The refactor work therefore started from a clean coverage baseline.
- **Templates packaged correctly**: `src/basanos/templates/` is inside the package root and `hatchling` includes all non-Python files by default — templates are present in the installed distribution without manual `package_data` configuration.

### Weaknesses

- **`_DiagnosticsMixin._iter_matrices` stub is an uncovered dead-code artefact** (`_engine_diagnostics.py:40`): The `raise NotImplementedError` body was added as a type-checker workaround during PR #287's typecheck fix (the refactored code had 18 `ty` errors). This is 1 of 2 uncovered statements. The stub is never executed — `BasanosEngine._iter_matrices` always overrides it. A `# pragma: no cover` annotation would be honest; a `Protocol`-based contract would be architecturally cleaner and eliminate the stub entirely.
- **`isinstance(data, dict)` guard re-introduced as dead code** (`_config.py:412`): Commit `8706d4b` explicitly removed this guard from `optimizer.py` on the grounds that Pydantic always passes a dict to `mode="before"` validators. The God Object refactor moved the validator to `_config.py` and the typecheck fix re-introduced the guard to satisfy `ty` (since `data` is annotated `object`). This is the second uncovered statement. The correct fix is to annotate `data` as `dict[str, object]` rather than guard with `isinstance`.
- **Jinja2 upper bound is a new tight ceiling**: `jinja2>=3.1,<4` follows the same pattern as the Polars ceiling that was just removed in entry #6. Jinja2 is stable and version 4.0 has no imminent release, but the `<4` upper bound will require a manual bump and sets a precedent that was just corrected for Polars.
- **`PortfolioData`/`Portfolio` inheritance couples the two in the MRO**: `Portfolio(PortfolioData)` means `isinstance(p, PortfolioData)` is true for `Portfolio` instances — potentially surprising to callers expecting a pure data container. If `Portfolio` ever needs a mutable computed field (e.g., a cached result), the frozen dataclass inheritance must be broken. An ADR documenting the intentional IS-A choice vs. composition would be worthwhile.
- **Mixin attribute contract is annotation-only, not enforced**: `_DiagnosticsMixin` and `_SignalEvaluatorMixin` declare `assets: list[str]`, `prices: pl.DataFrame`, and `mu: pl.DataFrame` as class-level annotations. These satisfy `ty` but provide no runtime enforcement — a subclass that forgets to provide these attributes would fail only at the first attribute access, not at instantiation. `Protocol` or `ABC` would make the contract formally verifiable.

### Risks / Technical Debt

- **No streaming / incremental API**: Unchanged from prior entries.
- **`_iter_matrices` stub creates an inconsistency in the class hierarchy**: `_DiagnosticsMixin._iter_matrices` raises `NotImplementedError` but is not decorated `@abstractmethod`, so `_DiagnosticsMixin` is not formally abstract. This means `isinstance(x, _DiagnosticsMixin)` is true for any `BasanosEngine`, but also any accidental direct instantiation of `_DiagnosticsMixin` would pass instantiation only to fail at method call time — the standard Python pitfall for non-ABC mixins.

### Score

**9/10** — Unchanged

**Rationale**: The session delivers four coherent structural improvements in a single day: God Object decomposition, `Portfolio` split, exception consolidation, and Jinja2 template migration. Each is technically correct and follows the established quality bar. The score does not increase because the two uncovered statements are direct artefacts of the typecheck fix strategy chosen during PR #287 (annotation-only mixin contract + `isinstance` guard), both of which would be addressed more cleanly with `Protocol`-based typing. The Jinja2 ceiling and `Portfolio` inheritance coupling are minor but repeat patterns that have required prior cleanup work. The codebase architecture is materially better today than at entry #6.

---

## 2026-03-21 — Analysis Entry #9

### Summary

Twelve commits since entry #8 close all three weaknesses and both risks from that session, then advance the architecture in two directions: (1) completing the `Portfolio` → composition migration with `PortfolioData` fully privatised and the double-construction eliminated (PRs #310, #314, #317), and (2) laying the infrastructure foundation for the long-requested streaming API via `StepResult` (PR #326) and `_StreamState` (PR #328) in `_stream.py`, with `StepResult` promoted to the public `basanos.math` namespace. The `cor` key type is narrowed to `dict[datetime.date, np.ndarray]` (`_engine_protocol.py:101`, PR #304). `_validate_inputs` extraction (PR #308) reduces `__post_init__` to a 3-line summary call. The `self: _EngineProtocol` pattern is documented (PR #313). Test count reaches 689 (up 33 from entry #8's 656). No new technical debt is introduced; the sole remaining structural gap is that `BasanosStream` itself — the step-loop consumer of `_StreamState` — does not yet exist.

### Strengths

- **`cor` key type narrowed correctly** (`_engine_protocol.py:101`, PR #304): `dict[object, np.ndarray]` → `dict[datetime.date, np.ndarray]`. The fix is applied at both the property and the Protocol declaration, so all three mixin consumers inherit the same precise contract. Code that indexes `cor[t]` can now be validated statically. This closes weakness #1 from entry #8 with one targeted annotation change.
- **`Portfolio` composition is clean and complete** (PRs #310, #314, #317): PR #310 eliminated `Portfolio(PortfolioData)` inheritance and replaced it with a `_data: PortfolioData` private field. PR #314 eliminated the double `PortfolioData` construction that was a performance artefact of the initial migration, and fully privatised the construction path. PR #317 declared `_data` as `ClassVar` to remove the sole remaining `type: ignore`. The three PRs together fully resolve the IS-A coupling concern filed across entries #7 and #8 — no `isinstance(p, PortfolioData)` surprise remains.
- **`_validate_inputs` extraction complete** (PR #308, `optimizer.py`): The six validation checks (date column, shape, column parity, non-positive prices, excessive NaNs, warm-up gap) now live in a `_validate_inputs` helper. `__post_init__` becomes a 3-line summary call. A `Warns` section documents the sliding-window warm-up boundary (PR #318). Weakness #3 from entry #8 is fully resolved.
- **Streaming API foundation is memory-efficient and mathematically correct** (`_stream.py:1–191`): `_StreamState` carries only 4×(1,N,N) + (N,N) + 8×(N,) arrays — O(N²) total, independent of T. This is in direct contrast to batch `_ewm_corr_numpy`, which holds 14×(T,N,N) arrays at peak (O(T·N²)). The IIR filter state reuse via `lfilter`'s `zi`/`zf` interface is the correct incremental implementation — the module docstring proves that the incremental path is bit-for-bit equivalent to the batch path. `StepResult` is already exported from `basanos.math`, signalling public API intent before `BasanosStream` ships.
- **`self: _EngineProtocol` pattern documented** (PR #313, `_engine_protocol.py`): The unconventional mixin `self` annotation is now explained in-situ. Risk #2 from entry #8 is closed.
- **EWMA+shrinkage vs. sliding-window covariance modes documented** (PR #320): The two covariance modes are now described with explicit trade-off language in the docstrings. New contributors no longer need to reverse-engineer the distinction from code.

### Weaknesses

- **`BasanosStream` class does not yet exist**: `_StreamState` and `StepResult` are the types on either side of `BasanosStream.step()`, but the class itself is absent. `StepResult` is already public — users importing it today have no entry point to produce one. The API is half-built: infrastructure committed, consumer class not yet written.
- **`StepResult.status` typed as `str` rather than `Literal`** (`_stream.py:188`): The four valid values (`"warmup"`, `"zero_signal"`, `"degenerate"`, `"valid"`) are documented and parametrically tested, but the field annotation is `str`. A `Literal["warmup", "zero_signal", "degenerate", "valid"]` annotation would enable type-checker exhaustiveness checking in `match`/`if` chains and flag invalid status strings at assignment sites. One-line fix.
- **Streaming tests are structural only** (`test_stream.py`, 15 tests): Tests verify shapes, dtypes, field names, mutability, and the public export — but no test calls any computation. There is no test that updating a `_StreamState` with a price row produces numerically correct filter state. The tests are necessary but insufficient for a production-ready streaming path.

### Risks / Technical Debt

- **Streaming API is infrastructure-only**: Two types exist, one is public. `BasanosStream` is absent. Any evolution of `_StreamState` (e.g., new fields for factor model state) before `BasanosStream` ships risks `test_stream.py::test_field_count` becoming a false constraint rather than a useful guard.
- **Marimo diagnostics notebook has no CI validation** (PR #302, `notebooks/`): The new notebook for `position_status`, `condition_number`, and `solver_residual` diagnostics is not executed in CI. The `make paper` target now compiles the LaTeX paper (resolving the prior concern about `.tex` without a build workflow), but there is no equivalent validation gate for notebooks.

### Score

**9/10** — Unchanged

**Rationale**: All entry #8 weaknesses and risks are resolved within 12 commits spanning one calendar day. The streaming API foundation is the most significant architectural advance since `_iter_solve` — if completed, it closes the longest-standing open item across all entries. The score does not advance because `BasanosStream` itself does not exist (the feature is half-built), `StepResult.status` has an imprecise type annotation, and the streaming test suite covers only structural invariants. These are addressable gaps, not architectural flaws. The `Portfolio` composition trilogy (PRs #310, #314, #317) is the cleanest multi-PR refactor in the journal to date.

---

## 2026-03-21 — Analysis Entry #10

### Summary

Seven commits since entry #9 complete the streaming API that has been the longest-standing open item across all journal entries. `BasanosStream` is now fully implemented: `from_warmup()` (PR #329), `step()` (PR #332), frozen semantics and warmup-status handling (PR #333), `StepResult.status` narrowed from `str` to `Literal` (PR #336), `WarmupState` + `warmup_state()` decoupling from `_iter_solve` (PR #339), and an EWM warmup short-circuit in `step()` that skips the O(N²) matrix reconstruction and O(N³) Cholesky solve during the pre-correlation period (PR #340). The streaming test suite grows from 15 structural tests to 48 tests covering numerical correctness (`rtol=1e-8` vs. batch engine), multi-step state continuity, state-invariant warmup assertions, timing verification, and a Hypothesis property suite. Total test count reaches 699. All three weaknesses from entry #9 are resolved.

### Strengths

- **`BasanosStream` closes the longest-standing structural gap** (`_stream.py:275`): Every journal entry since the initial analysis noted "No streaming / incremental API" as unchanged. That risk is now resolved. `from_warmup()` initialises the O(N²) IIR filter state from a historical batch; each subsequent `step()` advances by one row in O(N²) time. The O(N²) memory guarantee (4×(1,N,N) + (N,N) + 8×(N,) — independent of T) is proved in the module docstring and verified against the batch engine at `rtol=1e-8`.
- **EWM warmup short-circuit is correct and tested** (`_stream.py:623–650`, PR #340): When `state.step_count < cfg.corr`, `step()` updates all accumulators but returns early before the O(N²) correlation matrix reconstruction and O(N³) Cholesky solve. Three dedicated tests cover this: `test_stream_warmup_status_before_min_periods` (status label), `test_stream_warmup_skips_solve_state_unchanged` (asserts `profit_variance` and `prev_cash_pos` are unchanged across all warmup steps), and `test_stream_warmup_step_faster_than_full_solve` (asserts ≥1.5× timing advantage at N=20 over 30 steps).
- **`WarmupState` decoupling is architecturally correct** (`_engine_solve.py:30`, PR #339): `BasanosStream.from_warmup()` previously coupled to `BasanosEngine._iter_solve` private generator. PR #339 introduces `WarmupState` (frozen dataclass with `profit_variance` and `prev_cash_pos`) and `BasanosEngine.warmup_state()` as a public method. `from_warmup()` now calls `engine.warmup_state()` for these two scalars without touching the private generator.
- **`StepResult.status` is `Literal[...]`** (`_stream.py:208`, PR #336): `Literal["warmup", "zero_signal", "degenerate", "valid"]` enables static exhaustiveness checking in `match`/`if` chains, closes the one-line weakness from entry #9, and is consistent with the `position_status` column semantics in the batch engine.
- **`BasanosStream` frozen semantics are correctly implemented** (`_stream.py:331–333`): `__setattr__` raises `dataclasses.FrozenInstanceError` for any attribute write. The internal `_state: _StreamState` is mutable (enabling in-place accumulator updates without object churn per step), but the stream façade is immutable to callers. The asymmetry is intentional and documented.
- **Numerical correctness coverage is now comprehensive** (`test_stream.py:359–396`): `test_step_matches_basanos_engine` and `test_multi_step_matches_basanos_engine` both assert `np.testing.assert_allclose(..., rtol=1e-8, equal_nan=True)` against `BasanosEngine.cash_position`, closing the entry #9 weakness that tests were structural-only.

### Weaknesses

- **No `_StreamState` serialization interface** (issue #348): `_StreamState` is a plain mutable dataclass. It has no `__getstate__`/`__setstate__`, no pickle protocol, and no `save`/`load` helpers. A live system that crashes and restarts must call `from_warmup()` again on the full historical batch. For multi-year warmup windows this is a significant operational gap.
- **`from_warmup()` O(T·N²) redundancy** (issue #349): `from_warmup()` calls `BasanosEngine(prices, mu, cfg)` — which runs `_ewm_corr_numpy` internally — then immediately re-runs 4 independent `lfilter` sweeps over the same `T×N×N` arrays to extract final IIR `zf` states (`_stream.py:428–431`). The `warmup_state()` decoupling provides only `profit_variance` and `prev_cash_pos`; the IIR state is still re-derived from scratch. For T=2520, N=150 this doubles the O(T·N²) initialisation work.
- **`step()` missing input shape validation** (issue #350): `new_prices` and `new_mu` arrays are not validated against `(n_assets,)` before use (`_stream.py:556–561`). A wrong-length array produces a cryptic numpy error rather than a clear `ValueError`. This is the only unguarded public API boundary in the module.
- **SlidingWindowConfig streaming deferred with no timeline** (issue #356): `from_warmup()` raises `TypeError` for non-`EwmaShrinkConfig` inputs. `BasanosStream` and `StepResult` are public. Users who deploy `SlidingWindowConfig` for batch computation have no streaming path.

### Risks / Technical Debt

- **Warmup timing test is wallclock-based** (issue #351, `test_stream.py:584`): `test_stream_warmup_step_faster_than_full_solve` asserts a 1.5× speedup threshold via `time.perf_counter()`. The threshold is conservative but susceptible to CI scheduling noise. A mock-based approach (asserting `shrink2id` / `inv_a_norm` are not called during warmup) would be more robust.
- **No CI validation for Marimo notebook** (issue #355, unchanged from entry #9): The `notebooks/` directory is not executed in any workflow. Notebooks can accumulate stale cells without triggering CI failures.

### Score

**9/10** — Unchanged in number, but this entry marks the closure of the most structurally significant long-standing gap

**Rationale**: The streaming API is complete, correctly implemented, and comprehensively tested. Closing "No streaming / incremental API" — noted in every journal entry since the initial analysis — is the most material quality improvement in the codebase's history after the initial architecture. The score remains 9/10 rather than advancing to 10 because three concrete gaps prevent a perfect assessment: no serialization for `_StreamState` (a hard operational requirement for production systems), the O(T·N²) redundancy in `from_warmup()` (a performance concern for long warmup windows), and the missing input validation on `step()`. All three are tracked and addressable in targeted follow-up PRs.

---

## 2026-03-21 — Analysis Entry #11

### Summary

Seven commits since entry #10 close all four weaknesses and one of two risks from that session. The streaming API is now complete in every dimension: both covariance modes (`EwmaShrinkConfig` and `SlidingWindowConfig`) are fully supported, `from_warmup()` has no redundant O(T·N²) pass, `step()` validates its inputs, `BasanosStream` can be serialized and restored without re-running warmup, and the flaky wallclock timing test is replaced with a deterministic mock. Test count grows from 699 to 758 (+59). The sole remaining risk is the Marimo diagnostics notebook having no CI execution gate.

### Strengths

- **`from_warmup()` O(T·N²) redundancy eliminated** (`_ewm_corr.py`, `_engine_solve.py`, PR #349, commit `408b164`): `_ewm_corr_with_final_state()` is introduced alongside a new `_EwmCorrState` frozen dataclass. A single lfilter pass now yields both the full correlation tensor and the final IIR filter state (`corr_zi_x`, `corr_zi_xy`, `corr_zi_y`). `WarmupState` is extended with `corr_iir_state: _EwmCorrState | None`. `from_warmup()` extracts IIR state from `ws.corr_iir_state` directly — the four redundant lfilter sweeps are gone. The batch `_ewm_corr_numpy` is refactored as a one-line wrapper with zero code duplication. Coverage for both `_engine_solve.py` and `_ewm_corr.py` is 100%.
- **`BasanosStream.save()` / `load()` close the operational gap** (`_stream.py`, PR #348, commit `68e0862`): `np.savez` serialises all `_StreamState` numpy arrays; Pydantic `model_dump_json()` serialises `BasanosConfig`; the asset list is stored as a string array. `sw_ret_buf` uses an `np.empty((0,0))` sentinel to avoid `allow_pickle=True`. A restored stream produces bit-for-bit identical `step()` output. Three new tests verify round-trip correctness, asset/config preservation, and automatic `.npz` suffix append.
- **`SlidingWindowConfig` streaming is fully supported** (`_stream.py`, PR #356, commit `2931cb2`): The `TypeError` guard in `from_warmup()` is removed. `_StreamState` gains `sw_ret_buf: np.ndarray | None` — the `(W, N)` rolling buffer of vol-adjusted returns, seeded from the last W rows of `engine.ret_adj`. `step()` branches on config mode: EWM uses the existing IIR path; SW shifts the buffer in-place, fits `FactorModel.from_returns()` via truncated SVD, and solves via the Woodbury identity. The SW path includes `profit_variance` normalisation, degeneracy guards on both the FM solve and the denominator, and zero-signal early return — matching the batch engine exactly. `save()`/`load()` persist `sw_ret_buf`. 14 new tests cover all SW branches.
- **Input shape validation closes the unguarded public boundary** (`_stream.py:556–561`, PR #353, commit `a11463c`): `new_prices` and `new_mu` are now validated against `(n_assets,)` before any computation. Wrong-length arrays raise `ValueError` with a clear message rather than a cryptic numpy broadcast failure. This is the last unguarded public API boundary in the module.
- **Wallclock timing test replaced with deterministic mock** (`test_stream.py`, PR #354, commit `465585b`): `test_stream_warmup_step_faster_than_full_solve` used `time.perf_counter()` with a 1.5× threshold, susceptible to CI scheduling noise. The replacement asserts via `unittest.mock.patch` that `shrink2id` and `inv_a_norm` are not called during warmup steps — a structurally correct guard that cannot flap.
- **pytest-xdist enables parallel test execution** (`pyproject.toml`, `pytest.ini`, PR #347): `pytest-xdist` added to dev dependencies; `pytest.ini` configured with `-n auto`. The 758-test suite now distributes across available CPU cores, reducing CI wall time. Parallelisation is safe because no test has shared mutable global state.
- **EWM accumulator helper extracted and isolation tests added** (`_stream.py`, `test_stream.py`, PR #345): `_ewm_vol_accumulators_from_batch()` is extracted as a shared helper for the vol accumulator initialisation path in `from_warmup()`. 252 lines of new tests cover `from_warmup` state extraction in isolation — IIR/stream-state agreement, zero-signal rows, NaN-price rows, and all three exception paths in `warmup_state()`. Coverage for `_stream.py` reaches 100%.

### Weaknesses

- **`save()`/`load()` have no format version tag**: The `.npz` layout encodes field names as array keys but carries no schema version. If `_StreamState` gains or renames fields in a future PR, `load()` will either raise a `KeyError` (missing key) or silently ignore the new field, depending on which side changes. A `_FORMAT_VERSION = 1` constant written to the archive and asserted on load would make format evolution safe and diagnosable.
- **`FactorModel` component count is implicit in SW mode** (`_stream.py`, `FactorModel.from_returns()`): The number of SVD components used for the SW path is `min(n_assets, window - 1)` — a reasonable default but not a user-controllable parameter. Users with large universes where only a few factors dominate cannot reduce computational cost by specifying fewer components without extending `BasanosConfig`. The implicit truncation is undocumented in the public API surface.
- **BENCHMARKS.md baseline environment is three minor versions stale**: Entry #9 noted the benchmark environment shows "basanos version: 0.3.0" while the library is now at 0.4.2+. With the O(T·N²) redundancy elimination in `from_warmup()` representing a genuine 2× speedup for large warmup windows, the benchmark baseline is not only stale but actively misleading on the most recently optimised code path.
- **Marimo diagnostics notebook has no CI execution gate** (issue #355, unchanged from entry #10): The `notebooks/` directory accumulates cells referencing `position_status`, `condition_number`, and `solver_residual` diagnostics without any CI workflow running or validating them. With SW streaming now added to the public API, the notebook is at even higher risk of silent drift.

### Risks / Technical Debt

- **No CI validation for Marimo notebook** (unchanged from entry #10): Not resolved in this session. See weakness above.
- **SW `sw_ret_buf` NaN-padding in short-warmup edge case has no user warning**: When `from_warmup()` is called with fewer historical rows than the sliding window width, `sw_ret_buf` is padded with `np.nan` for the missing prefix rows. The first W `step()` calls will produce positions from a partially-NaN covariance matrix. The degeneracy guard will catch truly degenerate cases, but near-degenerate results pass through silently. A warning (or at minimum a `StepResult.status == "warmup"` return) during the padding-fill period would surface the data quality issue to callers.

### Score

**9/10** — Unchanged

**Rationale**: All four weaknesses and one of two risks from entry #10 are closed within a single development session. Both covariance modes now have complete, tested, serialisable streaming implementations. The `from_warmup()` redundancy elimination is a concrete 2× speedup for the initialisation path, not just a code-quality fix. The codebase is now at its strongest state across all 11 journal entries: 758 tests, 100% coverage, two complete streaming modes, crash-recovery serialisation, and no unguarded public API boundaries. The score does not advance to 10 because the `save()`/`load()` format version gap is a latent operational hazard, the BENCHMARKS.md baseline is materially stale against the newly optimised code path, and the Marimo notebook CI gap has persisted across three consecutive entries without resolution.

---

## 2026-03-21 — Analysis Entry #12

### Summary

Six commits since entry #11 close all three weaknesses and both risks from that session, and add one new targeted fix. `save()`/`load()` format versioning lands (`_SAVE_FORMAT_VERSION = 1`, PR #359). `max_components` is exposed on `SlidingWindowConfig` (PR #363) and immediately tightened with a cross-field validator (`max_components <= n_factors`, commit `e9f4945`, issue #369). SW NaN-padding warmup semantics are documented and tested (PR #364). BENCHMARKS.md is refreshed to v0.5.0 on Apple M4 Pro hardware with `from_warmup()` cases (PR #367). The longest-standing risk — Marimo notebook CI gap, open since entry #9 — is finally closed: `test_diagnostics_notebook.py` (350 lines, 36 tests) mirrors the `diagnostics.py` notebook's synthetic-data setup and exercises all five diagnostic properties across both covariance modes (PR #366). A `rhiza_marimo.yml` CI workflow independently discovers and executes all notebooks in `book/marimo/` via a dynamic matrix. Version is 0.5.0. Test count reaches 788.

### Strengths

- **Notebook CI gap closed on two levels** (PR #366, `da46f87`): `test_diagnostics_notebook.py` provides a pytest-integrated guard covering `position_status`, `condition_number`, `effective_rank`, `solver_residual`, and `signal_utilisation` for both EwmaShrink and SlidingWindow engines. Separately, `rhiza_marimo.yml` discovers and executes all `book/marimo/` notebooks in a GitHub Actions matrix. This two-layer approach means notebooks are validated as Marimo executables *and* their underlying API contracts are tested directly in the main pytest suite. The risk that has persisted across entries #9–#11 is now fully resolved at both layers.
- **`_SAVE_FORMAT_VERSION` closes the save/load operational hazard** (`_stream.py:73`, PR #359): `load()` raises a clear `ValueError` before any deserialization if the archive is missing the version key or the version doesn't match. The guard is applied at the top of `load()`, before any numpy arrays are read. Three tests cover: version present and correct, missing key, wrong version.
- **`max_components` exposed and immediately validated** (`_config.py`, PRs #363 + fix `e9f4945`): The field is added with a `gt=0` constraint and documented with the full `k_eff = min(n_factors, window, n_valid_assets[, max_components])` formula in LaTeX. Within the same session, a `model_validator(mode="after")` rejects `max_components > n_factors` at construction time — a misconfiguration that would otherwise be accepted silently and have no effect. Boundary case `max_components == n_factors` is accepted. Four tests cover all cases.
- **SW NaN-padding warmup contract formally stated** (`_stream.py`, PR #364): `from_warmup()` gains a `Notes` section proving that `sw_ret_buf` contains only real data by the time the first non-warmup `step()` result is returned. `step()` gains an inline comment documenting the EWM and SW warmup extension paths side by side. Two new tests verify the warmup-to-transition sequence and `save()`/`load()` round-trip in mid-warmup state. Risk #2 from entry #11 is resolved.
- **BENCHMARKS.md current against v0.5.0** (PR #367): Environment updated to Apple M4 Pro, version to 0.5.0. Five `from_warmup()` benchmark cases (EWM×3, SW×2) are regression-gated in CI. The prior baseline predated the 2× `from_warmup()` speedup; the new baseline makes that improvement visible.
- **788 tests** — up from 758 in entry #11. The 30-test increase is attributable to `test_diagnostics_notebook.py` (36 new tests) plus the `max_components` validator tests (2 new), offset by minor removals elsewhere. Test count growth is coherent with feature addition.

### Weaknesses

- **Benchmark baseline hardware mismatch**: The new BENCHMARKS.md baseline was captured on a local Apple M4 Pro (14-core, macOS). CI benchmark regression runs execute on GitHub Actions Ubuntu runners (typically 2–4-core x86 VMs). The M4 Pro produces significantly lower wall times, meaning CI will always run 2–4× slower than the baseline. The regression threshold (150%) may be too tight for normal CI scheduling variance on slower hardware. This creates a risk of false-positive benchmark failures unrelated to actual regressions.
- **`test_diagnostics_notebook.py` mirrors rather than imports notebook logic**: The test file re-implements the synthetic data generation constants and cell logic from `diagnostics.py` (same seed, same asset list, same mu construction). Any change to the notebook's data-generation cells that is not reflected in the test file will cause the gate to silently test a different scenario than the notebook runs. A `marimo run --output json` execution approach would test the notebook directly; the current approach tests the API the notebook uses.
- **Four other Marimo notebooks have no equivalent test gate**: `factor_model_guide.py`, `ewm_benchmark.py`, `shrinkage_guide.py`, and `demo.py` are covered by `rhiza_marimo.yml` (execution check only) but have no pytest-based API contract tests. `diagnostics.py` is now the only notebook with deep test coverage.

### Risks / Technical Debt

- **`rhiza_marimo.yml` execution vs. API contract**: The workflow-level check (marimo execute) will catch import errors and crashes, but not semantic drift — if a notebook cell produces a wrong value, the workflow passes. The pytest gate in `test_diagnostics_notebook.py` provides the semantic layer for `diagnostics.py` only. Extending this pattern to the other four notebooks would complete the protection.

### Score

**9/10** — Unchanged in number; the last persistent structural risk is closed

**Rationale**: The Marimo notebook CI gap, noted in every entry from #9 through #11 without resolution, is now closed. The `_SAVE_FORMAT_VERSION` guard eliminates the last operational hazard in the streaming serialization path. The `max_components` cross-field validator is an example of a weakness being caught and fixed in the same session it was introduced — exactly the right engineering discipline. The score does not advance to 10 because the benchmark baseline hardware mismatch is a methodology concern that will produce CI friction, `test_diagnostics_notebook.py` mirrors rather than executes the notebook (a subtle but real coupling gap), and the four remaining notebooks lack API-level test gates. The codebase is in its best state across all 12 entries.

---

## 2026-03-21 — Analysis Entry #14

### Summary

Five commits since entry #13 close two of the three weaknesses and the single risk from that session, and expose one new short-lived API instability in the process. `plan.md` is removed from the repository root (`76e66fa`). `MatrixBundle` is introduced to future-proof `_compute_position` against argument-list growth (`3ef3990`, PR #413). A conda-forge recipe is added, making basanos installable via `conda install -c conda-forge basanos` (`c97ccd0`, PR #407). The `ewm_corr` API goes through a rapid public/private flip: `edad633` promotes it as a proper public function (closes #410), then `8554b9a` immediately re-privatizes it after the promotion broke the benchmark notebook import. The notebook import fix required a separate commit on the open branch (`a34a29e`). The four remaining notebook suites' semantic coverage gap — weakness #2 from entry #13 — remains open.

### Strengths

- **`plan.md` artifact removed** (`76e66fa`): The 348-line agent planning artifact at the repository root, flagged in entry #13, is deleted in a focused `chore` commit. The weakness is closed. Repository root is clean.

- **`MatrixBundle` closes the `_compute_position` argument-list risk** (`_engine_solve.py:56–73`, PR #413): `MatrixBundle` is a frozen dataclass wrapping `matrix: np.ndarray`. `_compute_position` now takes `MatrixBundle | None` instead of a bare `np.ndarray | None`, and `_iter_matrices` yields `(i, t, mask, MatrixBundle | None)` via the existing `MatrixYield` alias. Any future covariance mode (DCC-GARCH, RMT-cleaned) that needs to carry auxiliary state alongside the covariance matrix can extend `MatrixBundle` with new fields without touching the method signature. `MatrixBundle` is re-exported from `basanos.math.__init__`. The risk from entry #13 is closed at the architecture level.

- **conda-forge recipe** (`recipe/meta.yaml`, PR #407): A complete conda-forge recipe is committed. The `noarch: python` build covers Python ≥ 3.11, all runtime dependencies are pinned consistently with `pyproject.toml` (numpy `>=2.4,<3`, polars `>=1.37.1`, scipy `>=1.17.0,<2`, pydantic `>=2.12.5,<3`, plotly `>=6.6.0,<7`). README gains a `conda install -c conda-forge basanos` install section. This is a meaningful distribution milestone: it enables users in scientific Python environments to install basanos without pip, expanding the addressable audience.

### Weaknesses

- **`ewm_corr` API flip introduced a broken state**: `edad633` promoted `ewm_corr` to a proper public function and exported it from `basanos.math.__init__`. `8554b9a` immediately re-privatized it without updating `book/marimo/notebooks/ewm_benchmark.py`, leaving the notebook with a broken import (`_ewm_corr_numpy` does not exist). This required a third commit (`a34a29e`) to fix the notebook. The two-commit reversal sequence — promotion followed by re-privatization within the same session — suggests the public API decision was not settled before the first commit landed. The final state (private) is correct given there is no documented external use case, but the churn left a broken intermediate state in the repository.

- **Four notebook test suites still validate execution, not semantics**: `test_demo_notebook.py`, `test_ewm_benchmark_notebook.py`, `test_factor_model_notebook.py`, and `test_shrinkage_notebook.py` each run `app.run()` and assert on exit codes, but do not validate computed values. Semantic drift (a notebook cell producing a wrong position or a silently changed parameter) passes all four suites. Only `test_diagnostics_notebook.py` asserts on actual computed values. This weakness carries forward from entry #13.

- **conda-forge recipe version is pinned to 0.5.0 in-tree**: `recipe/meta.yaml` hardcodes `version: 0.5.0` and the corresponding PyPI SHA256. There is no automation to keep this in sync with `pyproject.toml` on version bumps. As-is, every release requires a manual update to `recipe/meta.yaml` or the recipe drifts.

### Risks / Technical Debt

- **Polars upper bound absent from conda recipe**: `pyproject.toml` specifies `polars>=1.37.1,<2` but `recipe/meta.yaml` specifies only `polars >=1.37.1` (no upper bound). The conda and pip install paths will resolve different version ranges. If polars 2.x ships with breaking changes, conda users hit them silently while pip users are protected. The two specs should be kept in sync.

### Score

**9/10** — Unchanged

**Rationale**: `MatrixBundle` is a clean, forward-looking design move that directly addresses the risk raised in entry #13. The conda-forge recipe is a genuine distribution milestone. Removing `plan.md` closes the one cosmetic weakness from that entry. The score does not advance because the `ewm_corr` API flip exposed a gap in how API stability decisions are validated before merging — a reversal within the same session that left the repository in a broken intermediate state — and the four notebook test suites still lack semantic assertions. The recipe version drift and Polars upper-bound inconsistency are new, low-severity risks that add to the ongoing maintenance surface.

---

## 2026-03-21 — Analysis Entry #13

### Summary

Nineteen commits since entry #12 close all three weaknesses and the single risk from that session, then shift focus to internal code quality and test coverage. The benchmark baseline is recaptured on the GitHub Actions Ubuntu runner itself (`9cb8fee`, PR #371), eliminating the hardware-mismatch false-positive risk. The CI threshold is widened to 200% (`faedc28`, PR #381) as a complementary guard. `test_diagnostics_notebook.py` is rewritten to drive the notebook via `app.run()` rather than mirroring constants (`65eb866`, closes #379). Direct `app.run()` execution gates are added to all five notebook test files (`a5a6963`, closes #373), and the four remaining notebooks gain full pytest-based API contract suites (`3cb17b9`, closes #370). On the internals side: magic status strings are replaced by a `SolveStatus` StrEnum exported from the public API (`8f2e2b7`, PR #384); three static helpers (`_compute_mask`, `_check_signal`, `_scale_to_cash`) are extracted from `_SolveMixin`; `MatrixYield`/`SolveYield` type aliases annotate generator return types (`e9f6b63`, PR #386); `_compute_mask` calls are deduplicated across `_iter_solve` and `_iter_matrices` (`bb39d77`, PR #393); mask/signal/denom boilerplate is eliminated from `_iter_solve` (`0d4205b`, PR #391); and finally `_iter_solve` is refactored to delegate to `_iter_matrices`, with a new `_compute_position` static method holding the shared solve step (`a8b7b35`, PR #401). A 423-line numerical stability test suite is added covering near-singular matrices, `min_corr_denom` boundary semantics, ill-conditioned covariance, the `IllConditionedMatrixWarning` threshold, and a Hypothesis property on solver residuals (`bc20ee9`, PR #405). License compliance CI gates against copyleft dependencies (`a60209a`, PR #388). Semgrep replaces the retired `p/numpy` registry with a local `.semgrep.yml` (`180c947`, PR #395). GitHub issue templates and a PR checklist template are added (`77af5ed`, `181162b`). Test count reaches 900 (+112 from 788 in entry #12).

### Strengths

- **All three entry #12 weaknesses closed in this session**:
  - *Benchmark hardware mismatch*: `9cb8fee` recaptures the baseline on a GH Actions Ubuntu runner (x86, 4-core) — the exact environment where regression checks execute. `faedc28` widens the threshold from 150% to 200% as a complementary guard against CI scheduling jitter. The M4 Pro baseline that was producing systematically optimistic numbers is retired.
  - *`test_diagnostics_notebook.py` mirroring*: `65eb866` replaces all hardcoded constants and re-implemented cell logic with a `notebook_defs` module-scoped fixture that calls `app.run()` on `diagnostics.py` directly. Engine and config objects are extracted from the live notebook execution; zero-signal indices and asset counts are derived from engine state rather than hardcoded. The coupling gap between test and notebook is closed.
  - *Four remaining notebooks lacking API-level gates*: `3cb17b9` adds full pytest suites for `demo.py`, `ewm_benchmark.py`, `factor_model_guide.py`, and `shrinkage_guide.py` (705 lines, ~60 tests across four files). `a5a6963` adds a direct `app.run()` execution gate to every notebook test file, so the marimo runtime itself is exercised before any assertions run.

- **`SolveStatus` StrEnum closes the stringly-typed API surface** (`_engine_solve.py:31–52`, PR #384): `SolveStatus(StrEnum)` with members `WARMUP`, `ZERO_SIGNAL`, `DEGENERATE`, `VALID` replaces all magic string literals. `StrEnum` membership means `SolveStatus.VALID == "valid"` — backward compatibility with existing callers is preserved without a deprecation cycle. `SolveStatus` is re-exported from `basanos.math.__init__` and imported directly in `test_numerical_stability.py`, confirming it is a first-class public symbol.

- **`_iter_solve` delegates to `_iter_matrices`; `_compute_position` extracted** (`_engine_solve.py`, PR #401): `_iter_solve` is reduced from a 128-line method with duplicated covariance-dispatch logic to a ~25-line loop over `_iter_matrices`. The shared solve step — `inv_a_norm` denominator computation, `solve()` call, `SingularMatrixError` handling, delegation to `_denom_guard_yield` — is extracted into `_compute_position`, a static method with a full docstring and typed signature. Net change: 132 lines deleted, 69 added. Both covariance modes now reach the same code path; future changes to solve logic require a single edit.

- **Numerical stability test suite** (`test_numerical_stability.py`, PR #405): 423 lines covering five targeted scenarios and one Hypothesis property:
  1. Near-singular correlation matrices (condition number > 1e12) emit `IllConditionedMatrixWarning`.
  2. `min_corr_denom` boundary: denominator ≤ 1e-14 yields NaN, not a silent incorrect value.
  3. Ill-conditioned covariance with `shrink=1.0` and collinear assets: engine emits warning or returns `DEGENERATE`.
  4. `IllConditionedMatrixWarning` fires at exactly the documented 1e12 threshold, not 1e11 or 1e13.
  5. Singular correlation matrix returns `DEGENERATE` (not silent `NaN`).
  6. Hypothesis property: for every `VALID` step, `‖C·x − μ‖₂` is finite and bounded below a conservative tolerance.

- **License compliance CI gate** (`rhiza_licenses.yml`, PR #388): GitHub Actions workflow fails the build on any dependency with a GPL, LGPL, or AGPL license. `make licenses` target added for local replication. This is the first automated copyleft fence in the project.

- **Local Semgrep rules replace retired registry** (`.semgrep.yml`, PR #395): The `p/numpy` ruleset was retired by Semgrep. A 59-line local `.semgrep.yml` encoding the same numpy-safety rules now runs in CI via `rhiza_semgrep.yml`. The `make semgrep` target is added for local replication.

- **900 tests** — up from 788 in entry #12. The +112 delta is attributable to: four new full notebook suites (~60 tests), five new `app.run()` execution gates (5 tests), the numerical stability suite (~20 tests), and incidental additions from refactoring PRs. Growth is proportional to feature/coverage additions with no test churn.

### Weaknesses

- **`plan.md` committed to the repository** (commit `50a38e5`): A 348-line agent planning artifact is present at the repository root. This is not user-facing documentation, not referenced by any other file, and inconsistent with the project's documentation structure (ADRs, notebooks, `REPOSITORY_ANALYSIS.md`). It should either be moved to an `.ai/` or `docs/decisions/` directory or removed entirely.

- **Four notebook test suites test execution, not API contracts**: `test_demo_notebook.py`, `test_ewm_benchmark_notebook.py`, `test_factor_model_notebook.py`, and `test_shrinkage_notebook.py` all run `app.run()` and assert on `app.run()` return codes, but they do not validate the values produced by notebook cells. Semantic drift — a notebook computing a wrong position or a silently changed parameter — will pass the suite. Only `test_diagnostics_notebook.py` asserts on actual computed values.

- **`_ewm_corr_batch` is now an internal alias** (`_ewm_corr.py`, PR #376): `_ewm_corr_batch` was made public (`ewm_corr_batch`) and re-exported from `basanos.math.__init__` to fix a test import of a private symbol. The original private function is now a one-line wrapper. Exposing an internal computation helper as a public symbol expands the API surface without a documented use case or stability guarantee.

### Risks / Technical Debt

- **`_compute_position` is a static method calling instance-scoped dependencies via arguments**: The method takes `i, t, mask, expected_mu, matrix, denom_tol` explicitly and delegates to `_denom_guard_yield`, itself also static. The chain is clean for the current two-branch EWM/SW dispatch, but if a third covariance mode is added (e.g., DCC-GARCH), the argument list will grow unless the signature is reconsidered. This is a low-probability risk but worth noting as the architecture evolves.

### Score

**9/10** — Unchanged in number; all entry #12 weaknesses closed; internal quality meaningfully improved

**Rationale**: All three structural gaps from entry #12 are resolved. The `_iter_solve` refactor and `_compute_position` extraction complete a multi-PR deduplication campaign that began in entry #12 and produce the cleanest version of the core solve loop to date. The numerical stability suite closes a meaningful testing gap: previously, ill-conditioned matrix behavior was exercised only incidentally through integration tests; now it is tested directly with documented boundary conditions. The score does not advance to 10 because `plan.md` is noise at the repository root, the four newly added notebook suites validate execution but not semantics (a pattern that will mask silent regressions), and `ewm_corr_batch` is now a public symbol with no documented stability guarantee. The codebase is in its strongest state across all 13 entries.

---
