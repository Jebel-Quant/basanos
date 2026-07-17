## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

LOGO_FILE=.rhiza/assets/rhiza-logo.svg

# Override template default: include mkdocstrings plugin for API docs
MKDOCS_EXTRA_PACKAGES = --with 'mkdocstrings[python]'

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Test-layout parity: basanos groups tests by behaviour rather than mirroring
# the source tree 1:1, so a bespoke checker enforces the two properties that
# matter (every public module is tested; every test traces back to code) with
# the intentional behaviour-grouping recorded as configuration. Chained onto
# the CI `test` job (a double-colon rule) so a drift fails CI, not just review.
.PHONY: test-layout
test-layout: ## check public modules are tested and tests trace back to code
	@printf "${BLUE}[INFO] Checking test-layout parity${RESET}\n"
	@$(UV_BIN) run python scripts/check_test_layout.py

test:: test-layout

# Optional: developer-local extensions (not committed)
-include local.mk
