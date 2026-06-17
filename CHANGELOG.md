# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and entries are generated from [Conventional Commits](https://www.conventionalcommits.org).

## [0.7.3] - 2026-06-17

### New Features
- Migrate linalg to cvx-linalg dependency (#532)
- Subclass cvx.linalg linalg errors in basanos.exceptions

### Bug Fixes
- Re-export linalg errors from basanos.exceptions, not cvx.linalg

### Documentation
- Align README badges with jquantstats style
- Update SECURITY.md for basanos 0.7.x

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group across 1 directory with 4 updates (#536)
- Chore(deps)(deps): bump github/codeql-action in the github-actions group (#534)
- Bring coverage to 100%
- Update rhiza to v0.15.2 (#542)
- Update rhiza to v0.15.3 (#543)
- Update rhiza to v0.17.0 (#544)
- Update rhiza to v0.18.4 (#545)
- Chore(deps)(deps): bump the python-dependencies group with 3 updates (#547)
- Chore(deps)(deps): bump the github-actions group with 9 updates (#546)
- Chore(deps)(deps): bump the github-actions group with 8 updates (#548)
- Chore(deps)(deps): bump the python-dependencies group with 4 updates (#549)
- Sync rhiza template to v0.18.8 (#550)
- Chore(deps)(deps): bump the python-dependencies group with 5 updates (#552)
- Add Rhiza Claude commands (/rhiza_quality, /rhiza_update) (#553)
- Chore(deps)(deps): bump the python-dependencies group with 5 updates (#554)

### Other Changes
- Retire cholesky_solve
- Sync Rhiza template v0.18.8 → v0.19.3 (#555)
- Configure mkdocstrings paths=[src] so the book builds without editable install (#556)

## [0.7.2] - 2026-05-18

### Documentation
- Update CHANGELOG for jquantstats 0.9.1 bump
- Migrate README report.save() to report.to_html(path=...)
- Update CHANGELOG to mention README migration

### Maintenance
- Chore(deps-dev)(deps-dev): bump marimo in the python-dependencies group (#525)
- Chore(deps)(deps): bump github/codeql-action in the github-actions group (#524)
- Chore(deps)(deps): bump github/codeql-action in the github-actions group (#531)

### Other Changes
- Add PyPI monthly downloads badge to README (#527)
- Add CodeFactor badge to README
- Reduce math-engine complexity hotspots and remove subprocess-based notebook execution in tests (#529)
- Bump version 0.7.1 → 0.7.2

## [0.7.1] - 2026-05-03

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group with 3 updates (#522)

### Other Changes
- Feat/ic horizon parameter (#523)
- Bump version 0.7.0 → 0.7.1

## [0.7.0] - 2026-04-25

### New Features
- Add basanos.analytics shim forwarding to jquantstats (#521)

### Other Changes
- Bump version 0.6.5 → 0.7.0

## [0.6.5] - 2026-04-25

### Documentation
- List individual notebooks and reports in mkdocs nav
- Clean up mkdocs nav structure
- Fix math rendering — switch from RST to Markdown math syntax
- Replace RST :attr:/:meth: cross-refs with plain backtick code
- Replace all RST cross-reference syntax with plain backtick code

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group with 2 updates (#519)

### Other Changes
- Modify mkdocstrings configuration in mkdocs.yml
- Update template.yml to ref v0.10.3 and modify templates (#520)
- Bump version 0.6.4 → 0.6.5

## [0.6.4] - 2026-04-20

### Bug Fixes
- Point coverage badge at GitHub Pages URL
- Correct coverage badge link to html-coverage path

### Dependencies
- *(deps)* Lock file maintenance (#517)

### Other Changes
- Rhiza/v0.10.1 (#518)
- Bump version 0.6.3 → 0.6.4

## [0.6.3] - 2026-04-14

### Bug Fixes
- *(docs)* Add mkdocstrings[python] to MkDocs extra packages

### Documentation
- Simplify mkdocs.yml via INHERIT from docs/mkdocs-base.yml

### Dependencies
- *(deps)* Lock file maintenance (#511)
- *(deps)* Lock file maintenance (#514)

### Maintenance
- Chore(deps)(deps): bump docker/login-action in the github-actions group (#512)
- Chore(deps)(deps): bump pydantic in the python-dependencies group (#515)

### Other Changes
- Update template.yml to reference version v0.9.2 (#513)
- Rhiza/v0.9.5 (#516)
- Bump version 0.6.2 → 0.6.3

## [0.6.2] - 2026-04-02

### Dependencies
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.11.3 (#506)
- *(deps)* Update astral-sh/setup-uv action to v8 (#509)
- *(deps)* Update pre-commit hook rhysd/actionlint to v1.7.12 (#508)
- *(deps)* Lock file maintenance (#510)
- *(deps)* Update pre-commit hook jebel-quant/rhiza-hooks to v0.3.2 (#507)
- *(deps)* Update dependency astral-sh/uv to v0.11.3 (#505)

### Other Changes
- Remove security from central Makefile
- Delete REPOSITORY_ANALYSIS.md
- Bump version 0.6.1 → 0.6.2

## [0.6.1] - 2026-04-02

### Documentation
- *(changelog)* Backfill entries for v0.2.4 through v0.6.0
- *(paper)* Set author to Thomas Schmelzer <thomas@jqr.ae>
- *(paper)* Remove "Basanos:" prefix from title
- Add 2026-03-27 repository analysis entry
- Add 2026-03-27 repository analysis entry #2
- Add 2026-03-27 repository analysis entry #3

### Dependencies
- *(deps)* Update github/codeql-action action to v4.35.1 (#498)
- *(deps)* Update astral-sh/setup-uv action to v8 (#503)
- *(deps)* Update dependency astral-sh/uv to v0.11.3 (#502)

### Maintenance
- Run link-check on every push, typecheck across all supported Python versions
- Integrate typecheck with CI workflow
- Update typecheck command in CI workflow
- Update stale comment referencing removed rhiza_typecheck.yml
- Chore(deps)(deps): bump the python-dependencies group with 2 updates (#500)
- Apply rejected patch hunks and clean up .rej files

### Other Changes
- Add condition-number check on `factor_covariance` in `FactorModel.solve()` (#482)
- [WIP] Refactor README by splitting deep-dive content into docs (#483)
- Fix stale SECURITY.md support table and add deprecation policy + helper (#484)
- Extract shared EWM math primitives to eliminate streaming/batch duplication (#485)
- Initial plan (#489)
- Initial plan (#490)
- Enforce _EngineProtocol mixin contract via tests, docs, and stricter ty rules (#486)
- Add schema validation to BasanosStream serialisation (#492)
- Copilot/extract shared math primitives (#497)
- Add `woodbury_condition_number` diagnostic property to `FactorModel` (#495)
- Add cross-mode validation: EWMA vs. factor-model consistency tests (#496)
- Update template.yml to reference v0.8.20 (#504)
- Bump version 0.6.0 → 0.6.1

## [0.6.0] - 2026-03-27

### Documentation
- Migrate from pdoc/minibook to MkDocs (#455)
- *(paper)* Update basanos.tex — remove stale params, add IC metrics and ellipsoid figure
- *(paper)* Fix introduction roadmap — add missing sections 7 and 8
- *(paper)* Clarify EWMA correlation uses n(n+1)/2 upper-triangular pairs
- *(paper)* Fix stats.sharpe() and stats.max_drawdown() calls in listing
- Add uv sync to CONTRIBUTING, link from README, surface good first issues (#466)
- Add per-class navigable API reference via mkdocstrings (#468)
- Reduce repetition in README flow
- *(paper)* Restructure basanos.tex flow

### Dependencies
- *(deps)* Update actions/checkout action to v6 (#470)
- *(deps)* Update astral-sh/setup-uv action to v7 (#472)
- *(deps)* Update dependency astral-sh/uv to v0.11.2 (#469)
- *(deps)* Update actions/deploy-pages action to v5 (#471)
- *(deps)* Update actions/checkout action to v6 (#475)

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group with 5 updates (#451)
- Remove unused "deploy-docs" workflow step from rhiza_docs.yml
- Add smoke-test for the paper §6 code listing

### Other Changes
- Mkdocs (#458)
- Mkdocs (#459)
- Enforce exact symmetry on EWM correlation matrix estimates (#454)
- Delete .rhiza/tests/integration/test_marimushka.py
- Add .github/release.yml changelog config (#465)
- Add link-check.yml workflow (#464)
- Fix broken coverage badge link (#474)
- Add GitHub Discussion templates: help-wanted and ideas (#463)
- Add PyPI badge to README and expand CI matrix to Ubuntu + macOS (#467)
- Bump version 0.5.1 → 0.6.0

## [0.5.1] - 2026-03-24

### New Features
- Add format version tag to BasanosStream.save()/load() (closes #359)
- Add `max_components` to `SlidingWindowConfig` to expose and document SVD truncation (#363)
- Add CI execution gates for all four remaining Marimo notebooks (closes #370)
- Add license compliance workflow to block copyleft dependencies (#388)
- Apply max_turnover constraint in BasanosStream.step() (#436)

### Bug Fixes
- Document and test SW NaN-padding warmup semantics (closes #362) (#364)
- Validate max_components <= n_factors in SlidingWindowConfig (closes #369)
- Drive diagnostics notebook tests via app.run() instead of mirrored constants (closes #379)
- Promote ewm_corr to proper public function, drop private alias (closes #410)
- Add fill_nan(0.0) to position_delta_costs to clear EWMA warmup NaN (#435)

### Documentation
- Append 2026-03-21 analysis entry #12 to REPOSITORY_ANALYSIS.md
- Add 2026-03-24 post-analytics-refactor analysis entry

### Performance
- Recapture benchmark baseline on GitHub Actions Ubuntu runner (#371)

### Maintenance
- Ignore output/ directory
- Add direct notebook execution gate to all five notebook test files (closes #373) (#377)
- Add GitHub issue templates and Discussions template (#397)
- Eliminate duplicated mask/signal/denom boilerplate in `_iter_solve` (#391)
- Remove agent planning artifact plan.md from repo root
- Relax overly tight upper-bound dependency pins (#426)
- Add repository analysis entry #15 (2026-03-22)
- Add repository analysis entry #16 (2026-03-22)
- Add pytest API contract tests for end_to_end.py notebook (#441)
- Add weekly dep-compat CI job to detect API-breaking releases (#442)
- Document _iter_solve dual-path divergence and add cross-path consistency test (#444)
- Bring coverage to 100%

### Other Changes
- Sw construct (#367)
- [WIP] Add CI execution gate for Marimo diagnostics notebook (#366)
- [WIP] Fix test_ewm_benchmark_notebook.py imports of private symbol (#376)
- Initial plan (#382)
- Widen CI benchmark regression threshold from 150% to 200% (#381)
- Code quality: replace magic status strings with SolveStatus enum, extract shared helpers (#384)
- Add `MatrixYield`/`SolveYield` type aliases and annotate generator return types (#386)
- Consolidate duplicated `_compute_mask` calls in `_iter_solve` and `_iter_matrices` (#393)
- Replace retired Semgrep p/numpy registry ruleset with local rules file (#395)
- Plan
- [WIP] Add PR template with checklist requirements (#403)
- Add numerical stability test suite for basanos.math (#405)
- Refactor `_iter_solve`: extract `_compute_position`, delegate to `_iter_matrices` (#401)
- Add conda-forge recipe for basanos (#407)
- Introduce `MatrixBundle` to future-proof `_compute_position` against argument-list growth (#413)
- Re-privatize `ewm_corr` and fix test/notebook imports to use private module directly (#412)
- Consolidate BasanosEngine mixin chain into a single flat class (#427)
- Add numerical precision and regression tests against analytical solutions (#423)
- Add position-delta cost model and turnover budget constraint to BasanosEngine (#428)
- Vectorize EwmaShrink solve loop via batched numpy.linalg.solve (#429)
- Add end-to-end worked example notebook and fix NaN turnover during EWMA warmup (#430)
- Restore mixin inheritance in BasanosEngine to fix IDE Go-to-Definition (#438)
- Add license and repository information (#445)
- Update template.yml to reference version v0.8.16 (#446)
- [WIP] Remove profit variance EMA feature entirely (#447)
- Retire `basanos.analytics` subpackage — replace with `jquantstats` (#450)
- Bump version 0.5.0 → 0.5.1

## [0.5.0] - 2026-03-21

### New Features
- Add StepResult frozen dataclass as output type for BasanosStream (#326)
- Add `_StreamState` mutable dataclass carrying O(N²) IIR state (#328)
- Implement BasanosStream.from_warmup() classmethod (#329)
- Implement BasanosStream.step() for O(N²) incremental position update (#332)
- BasanosStream frozen semantics, warmup status, and required tests (#333)
- Add pytest-xdist for parallel test execution (#347)
- Add BasanosStream.save() / load() for crash-recovery without full re-warmup (closes #348)
- Implement SlidingWindowConfig streaming support in BasanosStream (closes #356)

### Bug Fixes
- Narrow `cor` key type from `dict[object, np.ndarray]` to `dict[datetime.date, np.ndarray]` (#304)
- Declare `_data` as `ClassVar` on `Portfolio` to remove `type: ignore` (#317)

### Documentation
- Append 2026-03-20 analysis entry #8 to REPOSITORY_ANALYSIS.md
- Document `self: _EngineProtocol` mixin pattern (#313)
- Add `Warns` section to `_validate_inputs` for sliding-window warm-up boundary (#318)
- Document EWMA+shrinkage vs. sliding-window factor model covariance modes (#320)
- Append 2026-03-21 analysis entry #9 to REPOSITORY_ANALYSIS.md

### Performance
- Eliminate O(T·N²) redundancy in BasanosStream.from_warmup() (closes #349)

### Dependencies
- *(deps)* Update github/codeql-action action to v4.34.0 (#301)
- *(deps)* Update github/codeql-action action to v4.34.1 (#346)

### Maintenance
- Extract `__post_init__` validation into `_validate_inputs` helper (#308)
- Eliminate double PortfolioData construction and privatise PortfolioData (#314)

### Other Changes
- Delete problem.md
- Split optimizer.py into _ewm_corr.py and _engine_solve.py sub-modules (#299)
- Add diagnostics Marimo notebook for position_status, condition_number, and solver_residual (#302)
- Refactor Portfolio to use composition instead of inheritance from PortfolioData (#310)
- Narrow `StepResult.status` from `str` to `Literal` type (#336)
- Decouple BasanosStream.from_warmup from BasanosEngine._iter_solve private API (#339)
- Short-circuit BasanosStream.step() during EWM warmup before O(N²) matrix solve (#340)
- Initial plan (#334)
- Add isolation tests for `from_warmup` state extraction and extract shared EWM accumulator helper (#345)
- Add input shape validation to BasanosStream.step() (#353)
- Replace wallclock timing assertion with mock-based early-return guard (#354)
- Analysis
- Bump version 0.4.3 → 0.5.0

## [0.4.3] - 2026-03-20

### New Features
- Add sliding_window benchmark cases and refresh BENCHMARKS.md baseline
- Increase test coverage to 100% and fix SingularMatrixError in cash_position
- Add position_status property to BasanosEngine to distinguish warmup/zero_signal/degenerate/valid rows
- Add migration validator and changelog for BasanosConfig covariance_config API change
- Export CovarianceConfig from basanos.math.__init__

### Bug Fixes
- Replace assert isinstance with cast to satisfy S101 linter rule
- Update TestSlidingWindowWarmupWarning to use new covariance_config API
- Remove dead-code k_eff < 1 branch from position_status
- Migrate benchmark fixtures to SlidingWindowConfig and refresh baseline
- Remove polars upper bound to eliminate manual version bump burden
- Annotate `_reject_legacy_flat_kwargs` as `dict[str, object]`, remove dead isinstance guard (#293)
- Replace annotation-only mixin contracts with Protocol-based typing (#294)

### Documentation
- Add ADR for Rhiza framework dependency
- Append 2026-03-19 analysis entry #2 to repository journal
- Append 2026-03-19 analysis entry #3 to repository journal
- Add 2026-03-20 repository analysis entry and benchmarks template
- Clarify EwmaShrinkConfig is intentionally minimal
- Append 2026-03-20 analysis entry #5 to REPOSITORY_ANALYSIS.md
- Append 2026-03-20 analysis entry #7 to REPOSITORY_ANALYSIS.md

### Dependencies
- *(deps)* Update dependency astral-sh/uv to v0.10.12

### Maintenance
- Remove dead-code k_eff < 1 branches
- Replace np.linalg.inv with _cholesky_solve in FactorModel.solve (Woodbury inner system)
- Promote singular-matrix edge case to deterministic unit test
- Add benchmarks to rhiza template
- Extend benchmark suite to cover all new Portfolio and BasanosEngine APIs
- Add cross-property consistency tests for position_status vs cash_position
- Eliminate DRY violation in position_status via _iter_solve generator
- Add dedicated fixtures for guaranteed position_status coverage
- Cover SingularMatrixError branch in `_iter_solve` solve() path (#270)
- Make `None` exclusive to warmup in `_iter_solve` sentinel contract (#272)
- Cover non-dict early-exit in `_reject_legacy_flat_kwargs` (#277)
- Cover sliding-window no-price degenerate path in _iter_solve (#278)
- Remove skip-based consistency tests superseded by dedicated position_status fixtures (#279)
- Cover sliding-window degenerate path and drop unreachable non-dict guard
- Remove example make modules and document active build system (#282)

### Other Changes
- Analysis
- Initial plan
- Merge pull request #219 from Jebel-Quant/copilot/update-rhiza-framework-dependency
- Initial plan
- Add make paper target for LaTeX compilation
- Merge pull request #221 from Jebel-Quant/copilot/add-tex-file-compilation-workflow
- Delete assets directory
- Initial plan
- Implement sliding window approach (Section 4.4) as alternative to EWMA/shrinkage
- Address code review: use typing.cast instead of assert, specific test match patterns
- Merge pull request #214 from Jebel-Quant/copilot/implement-sliding-window-approach
- Initial plan
- Narrow broad except Exception in _iter_matrices and cash_position to specific error types
- Merge pull request #227 from Jebel-Quant/copilot/narrow-except-exception-errors
- Initial plan
- Merge pull request #229 from Jebel-Quant/copilot/update-configuration-pattern
- Initial plan
- Merge pull request #226 from Jebel-Quant/copilot/add-benchmark-baseline-sliding-window
- Initial plan
- Fix absent paper PDF: add LaTeX build artifacts to .gitignore and harden paper.yml
- Merge pull request #231 from Jebel-Quant/copilot/fix-paper-pdf-issue
- Initial plan
- Apply ruff formatting fixes
- Merge branch 'main' into copilot/document-warm-up-gap-in-basanosconfig
- Merge pull request #225 from Jebel-Quant/copilot/document-warm-up-gap-in-basanosconfig
- Initial plan
- Merge pull request #239 from Jebel-Quant/copilot/refactor-remove-dead-code-k-eff
- Initial plan
- Merge pull request #235 from Jebel-Quant/copilot/refactor-solve-in-factor-model
- Merge pull request #245 from Jebel-Quant/renovate/astral-sh-uv-0.x
- Initial plan
- Merge pull request #236 from Jebel-Quant/copilot/feat-distinguish-cash-position-rows
- Initial plan
- Merge pull request #243 from Jebel-Quant/copilot/promote-singular-matrix-test
- Remove example benchmark tests
- Remove test_benchmarks
- Merge branch 'main' into sync
- Merge pull request #242 from Jebel-Quant/sync
- Initial plan
- Merge pull request #247 from Jebel-Quant/copilot/update-benmarks-version-to-v0-4-2
- Initial plan
- Merge pull request #246 from Jebel-Quant/copilot/update-migration-guide-for-basanosconfig
- Initial plan
- Merge pull request #254 from Jebel-Quant/copilot/export-covarianceconfig-type-alias
- Initial plan
- Merge pull request #255 from Jebel-Quant/copilot/add-docstring-to-ewmashrinkconfig
- Initial plan
- Document W-1 warm-up gap in SlidingWindowConfig.window field description
- Merge pull request #256 from Jebel-Quant/copilot/update-slidingwindowconfig-docs
- Initial plan
- Merge pull request #262 from Jebel-Quant/copilot/fix-remove-dead-code-position-status
- Initial plan
- Merge pull request #261 from Jebel-Quant/copilot/add-cross-property-consistency-tests
- Initial plan
- Fix lint: wrap long line and add match parameter to pytest.raises
- Merge pull request #264 from Jebel-Quant/copilot/fix-model-copy-pattern-fragility
- Initial plan
- Merge branch 'main' into copilot/refactor-cash-position-status-codes
- Merge pull request #263 from Jebel-Quant/copilot/refactor-cash-position-status-codes
- Initial plan
- Merge pull request #269 from Jebel-Quant/copilot/test-replace-skipped-consistency-tests
- Initial plan
- Merge pull request #268 from Jebel-Quant/copilot/fix-polars-upper-bound-burden
- Split Portfolio into PortfolioData + Portfolio analytics facade (#280)
- Consolidate exceptions.py: 25 → 18 exception types (#284)
- Adopt Jinja2 HTML templates for report generation (#288)
- Refactor BasanosEngine: decompose 83 KB God Object into focused sub-modules (#287)
- [WIP] Remove jinja2 upper bound to eliminate manual version bump (#292)
- [WIP] Update uv version to 0.10.12 for consistency (#295)
- [WIP] Fix malformed Issues URL in pyproject.toml (#296)
- Bump version 0.4.2 → 0.4.3

## [0.4.2] - 2026-03-19

### Bug Fixes
- Review pragma: no cover in portfolio.py — remove dead vol helper, add domain exceptions with descriptive messages, add tests
- Resolve ruff formatting issues in factor_model_guide.py
- Update coverage badge URL to gh-pages branch

### Maintenance
- Cover SingularMatrixError path in FactorModel.solve to reach 100% coverage

### Other Changes
- Initial plan
- Apply ruff format fixes
- Merge pull request #202 from Jebel-Quant/copilot/make-frozen-dataclass-for-factor-model
- Initial plan
- Fix ruff N806 lint: rename d_inv_B to d_inv_b_mat
- Merge pull request #204 from Jebel-Quant/copilot/implement-solve-method-factor-model
- Initial plan
- Merge pull request #184 from Jebel-Quant/copilot/review-no-cover-assertions
- Initial plan
- Merge pull request #208 from Jebel-Quant/copilot/create-marimo-notebook
- Initial plan
- Update README following recent commits (v0.2.3 features)
- Merge pull request #206 from Jebel-Quant/copilot/update-readme-file
- Initial plan
- Merge pull request #210 from Jebel-Quant/copilot/correct-coverage-badge
- Initial plan
- Merge pull request #212 from Jebel-Quant/copilot/increase-test-coverage
- Bump version 0.4.1 → 0.4.2

## [0.4.1] - 2026-03-19

### Other Changes
- Update ref version from v0.8.12 to v0.8.14
- Sync
- Sync
- Merge pull request #200 from Jebel-Quant/tschm-patch-1
- Bump version 0.4.0 → 0.4.1

## [0.4.0] - 2026-03-19

### New Features
- Add ConfigReport for BasanosConfig parameter analysis and lambda sweep
- Add test coverage badge via CI and genbadge
- Add structured JSON logging with JSONFormatter

### Bug Fixes
- Make Config Reports quick start example self-contained
- Use _prices/_mu in ConfigReport API reference example
- Fix coverage badge CI by using coverage.xml and genbadge[coverage]
- Update coverage-badge test to match xml report and genbadge[coverage] invocation
- Make analyse-repo fails - create REPOSITORY_ANALYSIS.md and fix tool permissions
- Catch SingularMatrixError in solver_residual and signal_utilisation
- Update coverage badge link to GitHub Pages HTML coverage report
- Add shell(make) permission to analyse-repo agent tool allowlist
- Only upload coverage report from primary Python version (3.12)

### Documentation
- Update README and add problem.md for trading cost and ConfigReport features
- Add link to basanos.pdf on paper branch in README
- Update BENCHMARKS.md and baseline.json for v0.3.0
- Add Attributes and Examples sections to BasanosEngine class docstring
- Add Marimo notebook index / catalog to docs/MARIMO.md
- Add Marimo notebook catalog to README, revert docs/MARIMO.md

### Dependencies
- *(deps)* Update actions/download-artifact action to v8
- *(deps)* Update actions/upload-artifact action to v7

### Maintenance
- Add coverage threshold fail_under = 90 to pyproject.toml
- Add explicit comments linking rhiza_typecheck.yml to Makefile hook and pyproject.toml config
- Add integration/smoke tests for full BasanosEngine pipeline
- Add explicit test for shrink=0 identity corner case (C=I → x=μ)
- Add edge-case tests for optimizer (zero mu, N=1, T=1, single non-null per row)
- Tighten dependency version bounds in pyproject.toml

### Other Changes
- Initial plan
- Add trading cost impact analysis to Portfolio, Plots, and Report
- Merge pull request #118 from Jebel-Quant/copilot/estimate-impact-of-trading-costs
- Initial plan
- Merge pull request #120 from Jebel-Quant/copilot/add-report-attribute-basanosconfig
- Initial plan
- Add companion paper in LaTeX (paper/basanos.tex + paper/basanos.bib)
- Merge pull request #122 from Jebel-Quant/copilot/write-companion-paper-latex
- Initial plan
- Add paper.yml workflow to compile LaTeX and publish PDF artifact
- Merge pull request #124 from Jebel-Quant/copilot/write-workflow-constructing-basanos-pdf
- Initial plan
- Update paper.yml to push basanos.pdf to paper branch via CI/CD
- Merge pull request #126 from Jebel-Quant/copilot/push-artifact-bolsane-pdf
- Initial plan
- Enhance paper: Rhiza section, outer product explanation, Boyd/Kahn/Schmelzer citation
- Merge pull request #128 from Jebel-Quant/copilot/enhance-paper-basanos-rhiza
- Initial plan
- Move Idea section below Table of Contents in README
- Merge pull request #130 from Jebel-Quant/copilot/move-idea-below-toc
- Initial plan
- Merge pull request #132 from Jebel-Quant/copilot/add-link-to-bosanos-pdf
- Initial plan
- Merge pull request #134 from Jebel-Quant/copilot/create-badge-for-test-coverage
- Initial plan
- Merge pull request #136 from Jebel-Quant/copilot/fix-coverage-badge
- Initial plan
- Merge pull request #138 from Jebel-Quant/copilot/fix-analyse-repo-failure
- REPO ANALYSIS
- Initial plan
- Merge pull request #146 from Jebel-Quant/copilot/add-coverage-threshold-to-pytest
- Initial plan
- Merge pull request #145 from Jebel-Quant/copilot/update-benchmarks-md-for-v030
- Initial plan
- Merge pull request #152 from Jebel-Quant/copilot/update-coverage-badge-url
- Initial plan
- Merge pull request #150 from Jebel-Quant/copilot/add-usage-examples-basanosengine-docstring
- Merge pull request #154 from Jebel-Quant/renovate/actions-download-artifact-8.x
- Merge pull request #155 from Jebel-Quant/renovate/actions-upload-artifact-7.x
- Initial plan
- Merge pull request #147 from Jebel-Quant/copilot/ci-fix-test-execution-error
- Initial plan
- Merge pull request #157 from Jebel-Quant/copilot/docs-add-marimo-notebook-index
- Initial plan
- Merge pull request #158 from Jebel-Quant/copilot/ci-confirm-type-checking
- Initial plan
- Merge pull request #156 from Jebel-Quant/copilot/add-integration-smoke-tests
- Initial plan
- Consolidate _MIN_CORR_DENOM and _MAX_NAN_FRACTION into BasanosConfig
- Merge pull request #167 from Jebel-Quant/copilot/arch-consolidate-module-constants
- Initial plan
- Merge pull request #170 from Jebel-Quant/copilot/add-test-for-shrink-zero-case
- Initial plan
- Merge pull request #171 from Jebel-Quant/copilot/add-edge-case-tests-for-optimizer
- Analysis
- Initial plan
- Merge pull request #174 from Jebel-Quant/copilot/improve-logging-structure
- Initial plan
- Merge pull request #185 from Jebel-Quant/copilot/tighten-dependency-version-bounds
- Initial plan
- Fix lint: use hyphen-minus, combine with statements, fix line length
- Merge pull request #186 from Jebel-Quant/copilot/document-silent-singularmatrixerror-handling
- Initial plan
- Add descriptive error messages to Portfolio.__post_init__() validation
- Merge pull request #187 from Jebel-Quant/copilot/add-error-messages-to-validation
- Initial plan
- Merge pull request #191 from Jebel-Quant/copilot/fix-upload-coverage-report
- Correct agent?
- Initial plan
- Initial plan
- Changes before error encountered
- Fmt
- Increase test coverage to 100%
- Merge pull request #189 from Jebel-Quant/copilot/increase-test-coverage-100
- Initial plan
- Update basanos.tex: add factor risk models, latent SVD factors, Sherman-Morrison, sliding window
- Merge pull request #199 from Jebel-Quant/copilot/update-basanos-discussion
- Bump version 0.3.0 → 0.4.0

## [0.3.0] - 2026-03-17

### New Features
- Add rolling Sharpe, rolling volatility, and annual breakdown
- Add IC, Rank IC, ICIR and related metrics to BasanosEngine
- Add temporal performance metrics to Stats (win_rate, profit_factor, payoff_ratio, monthly_win_rate, worst_n_periods, up/down capture)

### Bug Fixes
- Narrow Polars Series stat return types to satisfy typecheck
- Preserve date column when scaling cash positions in portfolio
- Round doctest output to avoid float precision mismatch
- Broaden metrics type annotation to resolve typecheck error
- Avoid scaling non-numeric columns in `cash_position` during portfolio creation
- Resolve typecheck errors in _report.py
- Create parent directories in Report.save before writing

### Documentation
- Add Report functionality to README

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group with 2 updates
- Update pytest configuration to exclude common directories
- Apply ruff formatting fixes
- Add 6 tests to reach 100% coverage

### Other Changes
- Merge pull request #92 from Jebel-Quant/dependabot/uv/python-dependencies-a6b05f733b
- Initial plan
- Implement turnover measures: Portfolio.turnover, turnover_weekly, turnover_summary, and shrinkage guide sweep
- Initial plan
- Add six diagnostic properties to BasanosEngine and tests
- Initial plan
- Initial plan
- Merge pull request #100 from Jebel-Quant/copilot/add-information-coefficient-metric
- Initial plan
- Merge pull request #94 from Jebel-Quant/copilot/implement-drawdown-measures
- Merge remote-tracking branch 'origin/main' into copilot/add-rolling-measures
- Merge pull request #104 from Jebel-Quant/copilot/add-rolling-measures
- Merge remote-tracking branch 'origin/main' into copilot/add-diagnostics-properties
- Merge pull request #102 from Jebel-Quant/copilot/add-diagnostics-properties
- Initial plan
- Merge branch 'main' into copilot/add-temporal-measures-metrics
- Merge pull request #98 from Jebel-Quant/copilot/add-temporal-measures-metrics
- Initial plan
- Apply ruff formatting fixes from pre-commit hooks
- Initial plan
- Apply ruff formatting fixes from pre-commit hooks
- Remove duplicate sharpe_at_shrink and naive_sharpe definitions
- Fix type checker errors and remove stale type: ignore comments
- Merge pull request #106 from Jebel-Quant/copilot/add-sharpe-ratio-function
- Merge branch 'main' into copilot/implement-turnover-measures
- Merge pull request #96 from Jebel-Quant/copilot/implement-turnover-measures
- Merge temporal measures metrics into README
- Initial plan
- Add detailed HTML report to Portfolio (monthly heatmap, Report class, tests)
- Merge pull request #108 from Jebel-Quant/copilot/make-detailed-html-report
- Initial plan
- Merge pull request #112 from Jebel-Quant/copilot/update-readme-report-functionality
- Initial plan
- Merge pull request #110 from Jebel-Quant/copilot/increase-test-coverage-to-100
- Initial plan
- Fix ruff formatting in property test files
- Merge pull request #114 from Jebel-Quant/copilot/implement-hypothesis-tests
- Update broken tests
- Bump version 0.2.4 → 0.3.0

## [0.2.4] - 2026-03-17

### Other Changes
- Add Idea section to README explaining the core concept
- Refine Idea section — remove 'naive', clarify C=I as Markowitz corner case
- Explain signal assessment purpose and factor orthogonalization in Idea section
- Missing project urls
- Backfill CHANGELOG.md
- Update dependency astral-sh/uv to v0.10.11
- Merge pull request #85 from Jebel-Quant/renovate/astral-sh-uv-0.x
- Bump version 0.2.3 → 0.2.4

## [0.2.3] - 2026-03-16

### New Features
- Add basanos demo marimo notebook
- Add cor_tensor property and flat-file round-trip tests for correlation matrices
- Remove pandas/pyarrow; replace with pure NumPy EWM correlation
- Optimize _ewm_corr_numpy with lfilter + add pandas vs NumPy comparison notebook
- Add data quality validation to BasanosEngine.__post_init__
- Enforce type safety with ty - add dev dependency, configuration, and CI workflow
- Add Cholesky-based solving and condition number monitoring for numerical stability
- Publish benchmark baseline results

### Bug Fixes
- Make cor_tensor doctest self-contained by defining engine fixture
- Export ewm_corr functions from app.setup without underscore prefix
- Fix lock file
- Resolve shrinkage_guide notebook cell failures
- Update SECURITY.md supported versions to match actual release v0.2.2

### Documentation
- Add justification and rationale for hardcoded defaults in BasanosConfig
- *(tests)* Improve docstring coverage across test files

### Dependencies
- *(deps)* Update astral-sh/setup-uv action to v7.4.0
- *(deps)* Update astral-sh/setup-uv action to v7.5.0
- *(deps)* Update actions/download-artifact action to v8.0.1
- *(deps)* Update dependency astral-sh/uv to v0.10.10
- *(deps)* Update softprops/action-gh-release action to v2.5.3
- *(deps)* Update softprops/action-gh-release action to v2.5.3
- *(deps)* Update dependency astral-sh/uv to v0.10.10
- *(deps)* Update softprops/action-gh-release action to v2.6.0
- *(deps)* Update softprops/action-gh-release action to v2.6.1
- *(deps)* Update dependency astral-sh/uv to v0.10.10
- *(deps)* Update astral-sh/setup-uv action to v7.6.0
- *(deps)* Update github/codeql-action action to v4.33.0

### Maintenance
- Convert basanos to editable local source in Marimo notebook
- Fix remaining ruff lint issue in test_correlation_matrices
- Add pytest fixture for test resources directory
- Use resource_dir fixture and resources/cor_tensor.npy for flat-file tests
- Consolidate test_analytics to mirror analytics module structure
- Consolidate test_math to mirror math module structure
- Add full coverage for _signal.py (shrink2id + vol_adj)
- Remove redundant TestCorTensorFlatFile test
- Remove redundant pytest.skip in asset3/asset4 NaN row test
- Add pandas and pyarrow as dev dependencies
- Reorder polars import to comply with style guidelines
- Remove unused `polars` import from ewm_benchmark.py

### Other Changes
- Merge branch 'main' into tschm-patch-1
- Merge pull request #10 from Jebel-Quant/tschm-patch-1
- Merge pull request #9 from Jebel-Quant/renovate/astral-sh-setup-uv-7.x
- Merge pull request #12 from Jebel-Quant/renovate/astral-sh-setup-uv-7.x
- Merge pull request #11 from Jebel-Quant/renovate/actions-download-artifact-8.x
- Merge pull request #13 from Jebel-Quant/renovate/astral-sh-uv-0.x
- Merge pull request #14 from Jebel-Quant/renovate/softprops-action-gh-release-2.x
- Delete src/basanos/main.py
- Delete tests/test_main.py
- Merge pull request #15 from Jebel-Quant/tschm-patch-1
- Update template.yml with new ref and exclusions
- Sync
- Merge remote-tracking branch 'origin/main' into tschm-patch-2
- Merge pull request #16 from Jebel-Quant/tschm-patch-2
- Merge pull request #18 from Jebel-Quant/renovate/softprops-action-gh-release-2.x
- Merge pull request #17 from Jebel-Quant/renovate/astral-sh-uv-0.x
- Add doctests to all source modules
- Add docstrings to reach 100% docs coverage
- Inline _compute_daily_profits_portfolio into Portfolio.profits
- Restore accidentally deleted src/basanos/__init__.py
- Merge pull request #19 from Jebel-Quant/docstrings
- Initial plan
- Merge pull request #21 from Jebel-Quant/copilot/make-notebook-with-basanos
- Merge pull request #22 from Jebel-Quant/NotebookBasanos
- Initial plan
- Merge pull request #26 from Jebel-Quant/copilot/test-correlation-matrices-construction
- Initial plan
- Merge pull request #28 from Jebel-Quant/copilot/store-matrices-in-tensor
- Initial plan
- Merge pull request #30 from Jebel-Quant/copilot/remove-pandas-and-pyarrow
- Merge pull request #33 from Jebel-Quant/renovate/softprops-action-gh-release-2.x
- Initial plan
- Remove hardcoded numbers from optimizer and add to BasanosConfig
- Merge pull request #32 from Jebel-Quant/copilot/remove-hardcoded-numbers
- Exclude rhiza.py from template configuration
- Remove rhiza.py from template.lock
- Delete book/marimo/notebooks/rhiza.py
- Merge pull request #34 from Jebel-Quant/renovate/softprops-action-gh-release-2.x
- Prepare for legal
- Bring in legal stuff
- Delete tests/benchmarks directory
- Merge pull request #35 from Jebel-Quant/copilot/remove-pandas-and-pyarrow
- Initial plan
- Address weaknesses: NaN Sharpe on zero vol, CHANGELOG.md, py.typed
- Merge pull request #37 from Jebel-Quant/copilot/address-weaknesses-issue
- Initial plan
- Merge pull request #39 from Jebel-Quant/copilot/fix-data-quality-validation
- Initial plan
- Add docstrings to exception __init__ methods to satisfy ruff D107
- Merge pull request #41 from Jebel-Quant/copilot/enhance-error-handling-functions
- Initial plan
- Add pre-commit hook to prevent Python cache file commits
- Merge pull request #48 from Jebel-Quant/copilot/fix-cache-pollution-issue
- Initial plan
- Fix date column dependency in Portfolio methods
- Merge pull request #46 from Jebel-Quant/copilot/fix-date-column-dependency
- Initial plan
- Merge branch 'main' into copilot/fix-hardcoded-assumptions
- Merge pull request #43 from Jebel-Quant/copilot/fix-hardcoded-assumptions
- Initial plan
- Add logging warning when normalisation denominator is degenerate
- Merge pull request #50 from Jebel-Quant/copilot/add-logging-for-normalization-guard
- Remove excluded paths from template.yml
- Optimizer
- Analysis
- Add basanos benchmark tests
- Merge pull request #51 from Jebel-Quant/tschm-patch-100
- Initial plan
- Fix zero-division guards: pre-check expected_mu, remove dead denom is None, suppress NaN divide warnings
- Merge pull request #53 from Jebel-Quant/copilot/fix-zero-division-guards
- Initial plan
- Merge pull request #55 from Jebel-Quant/copilot/improve-type-hint-coverage
- Initial plan
- Fix long docstring lines in exceptions.py to pass ruff E501 check
- Merge pull request #58 from Jebel-Quant/copilot/fix-error-context-loss
- Initial plan
- Merge pull request #62 from Jebel-Quant/copilot/improve-docstring-coverage
- Initial plan
- Fix line-length lint errors in TestProfitVarianceEMA tests
- Merge pull request #64 from Jebel-Quant/copilot/add-unit-tests-for-profit-variance
- Initial plan
- Add MissingDateColumnError and IntegerIndexBoundError exception classes for explicit date column handling
- Add IntegerIndexBoundError and fix exception messages to match tests
- Fmt
- Fix MissingDateColumnError message to match docstring
- Fix test match string to align with MissingDateColumnError message
- Merge pull request #60 from Jebel-Quant/copilot/fix-date-column-assumption
- Delete tests/benchmarks/test_benchmarks.py
- Initial plan
- Merge pull request #71 from Jebel-Quant/copilot/add-numerical-stability-safeguards
- Initial plan
- Merge pull request #70 from Jebel-Quant/copilot/publish-benchmark-results
- Initial plan
- Add hypothesis dev dependency for property-based testing
- Initial plan
- Apply ruff auto-fixes to property test files
- Merge pull request #72 from Jebel-Quant/copilot/add-property-based-tests
- Initial plan
- Apply ruff format fixes
- Merge pull request #69 from Jebel-Quant/copilot/document-shrinkage-methodology
- Initial plan
- Merge pull request #79 from Jebel-Quant/copilot/fix-security-md-version-references
- Initial plan
- Document optimizer performance cliffs: complexity, memory, and practical limits
- Merge pull request #73 from Jebel-Quant/copilot/document-optimizer-performance-cliffs
- Initial plan
- Add explicit is_positive_definite check for numerical stability
- Merge pull request #74 from Jebel-Quant/copilot/fix-numerical-stability-issue
- Merge pull request #82 from Jebel-Quant/renovate/astral-sh-uv-0.x
- Merge pull request #83 from Jebel-Quant/renovate/astral-sh-setup-uv-7.x
- Merge pull request #84 from Jebel-Quant/renovate/github-codeql-action-4.x
- Initial plan
- Add post-validate hook to run typecheck as part of make validate
- Merge pull request #80 from Jebel-Quant/copilot/add-type-checking-ci
- Remove REPO
- Bump version 0.2.2 → 0.2.3

## [0.2.2] - 2026-03-10

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group with 2 updates
- Update rhiza reference to v0.8.9
- Apply ruff auto-formatting to stats and tests

### Other Changes
- Merge pull request #4 from Jebel-Quant/dependabot/uv/python-dependencies-41f8a01e75
- Sync
- Merge pull request #7 from Jebel-Quant/tschmSync
- Initial plan
- Merge pull request #6 from Jebel-Quant/copilot/add-summary-method-to-stats
- Bump version 0.2.1 → 0.2.2

## [0.2.1] - 2026-03-10

### Other Changes
- Bump version 0.2.0 → 0.2.1

## [0.2.0] - 2026-03-10

### Dependencies
- *(deps)* Update package dependencies and rhiza template ref

### Maintenance
- Test trivial
- Tests
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps)(deps): bump plotly in the python-dependencies group
- Apply rhiza v0.12.1 sync updates

### Other Changes
- Initial commit
- Sync
- Lock file
- Add tests for main.py
- Update main.py
- Deps
- Add docstrings to test_main.py to satisfy ruff D100/D103
- Copy from taipan
- Rename taipan → optimizer, add professional README
- Comment out fig.show() in README to fix headless test
- Merge pull request #2 from Jebel-Quant/dependabot/github_actions/github-actions-8327ffd391
- Merge pull request #1 from Jebel-Quant/dependabot/uv/python-dependencies-698941b3fe
- Update reference to version v0.8.7
- Sync
- Merge pull request #3 from Jebel-Quant/tschm-patch-1
- Bump version 0.1.0 → 0.2.0

<!-- generated by git-cliff -->
