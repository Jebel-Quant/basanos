# CLAUDE.md

Guidance for working in this repository.

## What this is

`basanos` — correlation-aware portfolio optimization and analytics, part of the
[jebel-quant](https://github.com/jebel-quant) ecosystem. Numeric work is
`numpy`/`scipy` over `polars` frames, configuration is `pydantic`, and reporting
renders `jinja2` templates to `plotly` figures.

Everything lives under `src/basanos/`, and the package is deliberately
front-loaded with private modules — the public surface is small:

- `math/optimizer.py` — the entry point. The `_engine_*` modules behind it split
  one large solver into named concerns: `_engine_core`, `_engine_solve` and
  `_engine_solve_base` (the solve itself), `_engine_validation`,
  `_engine_diagnostics`, `_engine_ic`, `_engine_performance`, and
  `_engine_protocol` (the interface the others agree on).
- `math/_stream*.py` — the streaming/online path: `_stream`, `_stream_state`,
  `_stream_math`, `_stream_solve`, `_stream_io`.
- `math/_config.py`, `_config_report.py` — the pydantic config and its HTML
  report, rendered from `templates/config_report.html` on `templates/_base.html`.
- `math/_factor_model.py`, `_signal.py` — the factor model and signal handling.
- `analytics/` — the reporting shim; `exceptions.py`, `_logging.py` and
  `_deprecation.py` are the cross-cutting utilities.

Around the package: `benchmarks/` with `BENCHMARKS.md`, `book/` (marimo notebooks
under `book/marimo/notebooks`), `paper/`, `recipe/` for the conda feedstock, and
`scripts/`.

## Ownership: locally owned vs Rhiza-managed

This repo syncs its dev infrastructure from the
[`jebel-quant/rhiza`](https://github.com/jebel-quant/rhiza) template. The pinned
version lives in `.rhiza/template.yml` (`ref:`), and `/rhiza:update` re-applies
the template. **The authoritative, machine-generated list of synced files is the
`files:` block of `.rhiza/template.lock`** — when in doubt, consult it. The split
below summarises it.

### Locally owned — edit these freely

- `src/` — the library source, including the HTML templates it ships
- `tests/`, `benchmarks/` — the test suite and the benchmark suite
- `pyproject.toml` — project metadata, dependency groups, tool config, and the
  `[tool.rhiza-task]` table that configures the gates
- `README.md`, `BENCHMARKS.md`, `CHANGELOG.md`, `mkdocs.yml`, `CLAUDE.md`
- `book/`, `paper/`, `recipe/`, `scripts/` — notebooks, the paper, the conda
  feedstock recipe, and repo scripts
- `.rhiza/template.yml` — the template pin and the `profiles:`/`templates:`
  selection. The one file under `.rhiza/` this repo owns.
- `local.mk` — repo-specific make targets. The `Makefile` `-include`s it, and the
  template deliberately does not ignore it.

### Rhiza-managed — do NOT edit in place; fix upstream

These are overwritten by the next sync. To change one, open a PR against
`jebel-quant/rhiza` (or exclude the path in `.rhiza/template.yml`), then re-sync:

- `.github/workflows/rhiza_*.yml` — all CI/CD workflows
- `.github/` scaffolding — `dependabot.yml`, `release.yml`, rulesets,
  `secret_scanning.yml`. `CONFIG.md` is excluded in `template.yml` rather than
  synced.
- `Makefile` — a 71-line shim that pins `RHIZA_TASK` and forwards every unmatched
  target to that CLI. Nothing goes below it; the next sync overwrites whatever was
  appended. Repo targets belong in `local.mk`.
- `.pre-commit-config.yaml`, `ruff.toml`, `pytest.ini`, `.bandit`,
  `.editorconfig`, `.python-version`, `cliff.toml` — tooling config
- `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, and the synced `docs/` pages

`SECURITY.md` in particular is synced here: an edit to it is drift the next sync
reverts, and the `check-managed-files` pre-commit hook refuses the commit.

## Quality gates

Since rhiza v1.4 the gates are tasks in the pinned `rhiza-task` CLI rather than
synced make fragments. Run them as bare `make <target>` (the shim forwards to
`uvx rhiza-task <task>`) — never call `.venv/bin/...` directly. `make help` lists
every task the pinned CLI knows, plus anything `local.mk` adds.

- `make install` — create the venv and sync dependencies
- `make fmt` — the pre-commit hooks over all files
- `make typecheck` — `ty` **and** `mypy`, because `[tool.rhiza-task]` sets
  `typechecker = "both"`
- `make test` — the full pytest suite with the coverage gate
- `make coverage` — coverage measurement into `_tests/coverage.xml`
- `make docs-coverage` — interrogate docstring coverage
- `make deps` — deptry unused/missing dependency analysis
- `make security` — the bandit scan
- `make license` — fail on GPL/LGPL/AGPL
- `make rhiza-test` — the rhiza repository checks, from `pytest-rhiza==0.2.1`
- `make benchmark` — the performance benchmarks
- `make marimo` — the notebook editor, rooted at `book/marimo/notebooks`
- `make all` — the gate set CI runs

Do not reach for `make mutation`. The task still exists in the CLI, but rhiza
v1.5.0 stopped offering mutation testing (Jebel-Quant/rhiza#1492) and the recipe
drives a mutmut 2.x CLI that mutmut 3 removed.

## Conventions

- **The coverage threshold is set twice, and they are two settings, not one.**
  `[tool.coverage.report]`'s `fail_under = 90` is what `coverage report` uses;
  rhiza-task's own default is also 90, which is why `coverage-fail-under` is
  deliberately absent from `[tool.rhiza-task]`. Raising the bar means changing
  both — the CLI value outranks the `[tool.coverage.report]` one for the `test`
  task.
- `marimo-folder = "book/marimo/notebooks"` in `[tool.rhiza-task]` — this repo
  does not use rhiza's default notebook location, so the notebook tasks need it.
- `[tool.deptry.package_module_name_map]` maps distribution names to import
  names. A new dependency whose import name differs from its package name needs
  an entry, or `make deps` reports it as missing.
- The per-test timeout is 60s (`pytest-timeout`).
- Three markers are declared: `stress`, `property`, and `slow` for
  realistic-scale runs. Deselect with `-m "not slow"`. Use these rather than
  inventing new ones.
- The private-module convention is load-bearing: a leading underscore means "not
  part of the API". Promoting something to public means re-exporting it from
  `basanos/__init__.py` or `basanos/math/__init__.py`, deliberately.
  `_deprecation.py` exists so a rename can be staged rather than breaking callers.

## Test layout

Tests mirror the package's subpackages — `tests/test_math/`,
`tests/test_analytics/` — plus three groups that deliberately have no 1:1 source
counterpart:

- `tests/test_integration/test_engine_integration.py` — the engine end to end,
  rather than per `_engine_*` module.
- `tests/test_math/test_*_notebook.py` — one per notebook (`demo`,
  `diagnostics`, `end_to_end`, `factor_model`, `shrinkage`). These execute the
  notebooks, so a broken notebook fails the suite rather than only the book build.
- `tests/test_math/test_numerical_regression.py` and
  `test_numerical_stability.py` — pinned against the golden `.npy` fixtures in
  `tests/resources/` (`cor_tensor`, `golden_2asset_cash_position`,
  `golden_ewma_warmup_cash_position`). **Regenerate a golden file only with a
  deliberate, reviewed reason** — silently refreshing one turns a regression test
  into a tautology.

`test_optimizer_property.py` and `test_signal_property.py` hold the hypothesis
properties; `test_paper_example.py` keeps `paper/` honest;
`tests/benchmarks/test_basanos.py` is the timing suite. Shared fixtures live in
`tests/conftest.py`.
