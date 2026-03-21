# Issue: `rhiza_validate.yml` bypasses `make validate` — `post-validate` hook never fires in CI

## Summary

`rhiza_validate.yml` currently runs two separate steps:

1. `uvx "rhiza>=0.8.0" validate .`
2. `make rhiza-test`

This bypasses `make validate` entirely, so any `post-validate::` hook defined in a project's `Makefile` is **never executed in CI**.

## Current behaviour

In `rhiza_validate.yml`:

```yaml
- name: Validate Rhiza config
  if: ${{ github.repository != 'jebel-quant/rhiza' }}
  shell: bash
  run: |
    uvx "rhiza>=0.8.0" validate .

- name: Run Rhiza Tests
  shell: bash
  run: |
    make rhiza-test
```

Because `make validate` is never called, any `post-validate::` hook (e.g. `post-validate:: typecheck`) silently does nothing in CI, even though it runs correctly when a developer calls `make validate` locally.

## Expected behaviour

`rhiza_validate.yml` should call `make validate` so the full hook chain fires — just as it does locally.

## Suggested fix

Replace the two separate steps with a single `make validate` call:

```yaml
- name: Validate
  if: ${{ github.repository != 'jebel-quant/rhiza' }}
  shell: bash
  run: |
    make validate
```

`make validate` already calls `uvx rhiza validate .` internally (via `rhiza.mk`) and then fires the `post-validate::` hook, so this is a strict improvement with no loss of functionality.

## Context / discovered in

Discovered while working on [Jebel-Quant/basanos](https://github.com/Jebel-Quant/basanos). The project has:

```makefile
post-validate:: typecheck
```

This hook ran correctly locally (`make validate`) but was silently skipped in every CI run because `rhiza_validate.yml` called `uvx rhiza validate .` directly instead of going through `make validate`.
