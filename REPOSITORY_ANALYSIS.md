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

