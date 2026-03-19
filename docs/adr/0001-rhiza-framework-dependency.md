# 0001 — Rhiza Framework Dependency

**Status:** Accepted
**Date:** 2026-03-19

---

## Context

Basanos uses [Rhiza](https://github.com/jebel-quant/rhiza) as a development framework template. Rhiza provides a standardised set of:

- GitHub Actions CI/CD workflows (14 workflow files, all prefixed `rhiza_*`)
- A modular Makefile system (`.rhiza/rhiza.mk` and `.rhiza/make.d/*.mk`)
- Project scaffolding (pre-commit hooks, linting configuration, release tooling)
- Agentic workflow utilities

The project badge `Synced with Rhiza` reflects that these files are actively synced from the upstream template at `jebel-quant/rhiza`. The coupling is intentional — Rhiza evolves centrally and propagates improvements to all downstream projects through the `make sync` / `rhiza_sync.yml` workflow.

The concern raised is that this coupling creates a risk: if the Rhiza template repository changes substantially or is deprecated, a proportionally large rework of CI and Makefile infrastructure would be required.

---

## Decision

Continue using Rhiza as the development framework template. The benefits of standardisation and centralised maintenance outweigh the coupling risk, provided the following mitigations are in place.

**Mitigations already implemented:**

1. **Version-pinned sync.** `template.yml` pins the upstream at a specific tag (`ref: "v0.8.14"`). No automatic drift occurs; updates are opt-in via an explicit `make sync` run.

2. **Committed workflow files.** All `rhiza_*.yml` workflows are committed directly to the repository. CI does not fetch templates at runtime. A deprecation of the upstream template would leave all existing workflows fully functional.

3. **Self-contained `.rhiza/` directory.** The entire Rhiza toolchain lives in `.rhiza/` and is committed. The repository can be cloned, built, and tested with no external Rhiza dependency.

4. **Escape hatch documented** (see [Consequences](#consequences) below).

---

## Consequences

### Positive

- Consistent tooling, naming conventions, and CI pipelines across all Rhiza-managed projects.
- Central improvements (security patches, new tooling) propagate automatically when `make sync` is run.
- Reduced per-project boilerplate; the team focuses on domain logic rather than infrastructure.

### Negative / Risks

- **Coupling risk.** Substantial upstream changes require reviewing and applying updates across all synced files.
- **Namespace visibility.** The `rhiza_*` workflow prefix makes the framework origin explicit; renaming would require manual intervention.

### Deprecation / Exit Strategy

If Rhiza is deprecated or the team decides to remove the dependency, the following steps are sufficient:

1. **Freeze current state.** Stop running `make sync`. All workflow files, Makefile fragments, and configuration files already present in the repository continue to work indefinitely.
2. **Rename workflows** (optional). Strip the `rhiza_` prefix from workflow filenames and update any cross-references (e.g. the CI badge URL in `README.md`).
3. **Extract the Makefile.** Copy the relevant targets from `.rhiza/make.d/*.mk` into the root `Makefile` (or new files under a project-owned directory such as `make.d/`). Remove the `include .rhiza/rhiza.mk` line.
4. **Delete `.rhiza/`.** Once all targets and configuration have been migrated, the `.rhiza/` directory can be removed.

None of these steps require external access; the complete Rhiza source is already present in the repository.

---

## Alternatives Considered

| Alternative | Reason not chosen |
|-------------|-------------------|
| Copy-paste templates once and maintain independently | Loses the ability to receive upstream improvements; higher long-term maintenance burden |
| Use a generic CI template (e.g. GitHub reusable workflows) | Less opinionated; would require recreating the Makefile conventions and agentic workflow support from scratch |
| No template framework | Maximum flexibility but maximum per-project infrastructure cost |
