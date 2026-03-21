## Makefile (repo-owned)
# Keep this file small. It can be edited without breaking template sync.

DOCFORMAT=google
DEFAULT_AI_MODEL=claude-sonnet-4.5
LOGO_FILE=.rhiza/assets/rhiza-logo.svg
GH_AW_ENGINE ?= copilot  # Default AI engine for gh-aw workflows (copilot, claude, or codex)

# Always include the Rhiza API (template-managed)
include .rhiza/rhiza.mk

# Optional: developer-local extensions (not committed)
-include local.mk

## Hooks

# CI equivalent: .github/workflows/rhiza_typecheck.yml runs `make typecheck`
# on every push and pull_request to main/master.
post-validate:: typecheck ## run type checking as part of make validate

# CI equivalent: .github/workflows/notebooks.yml runs `make marimo-validate`
# on every push and pull_request that touches book/marimo/notebooks/.
post-validate:: marimo-validate ## validate Marimo notebooks as part of make validate

## Custom targets

##@ Paper

.PHONY: paper
paper: ## compile the LaTeX paper (paper/basanos.tex → paper/basanos.pdf)
	@command -v latexmk >/dev/null 2>&1 || { printf "${RED}[ERROR] latexmk not found. Install TeX Live (https://tug.org/texlive/) with your package manager.${RESET}\n"; exit 1; }
	@printf "${BLUE}[INFO] Compiling paper/basanos.tex...${RESET}\n"
	@cd paper && latexmk -pdf -interaction=nonstopmode basanos.tex
	@printf "${GREEN}[INFO] Paper compiled: paper/basanos.pdf${RESET}\n"

.PHONY: adr
adr: install-gh-aw ## Create a new Architecture Decision Record (ADR) using AI assistance
	@echo "Creating a new ADR..."
	@echo "This will trigger the adr-create workflow."
	@echo ""
	@read -p "Enter ADR title (e.g., 'Use PostgreSQL for data storage'): " title; \
	echo ""; \
	read -p "Enter brief context (optional, press Enter to skip): " context; \
	echo ""; \
	if [ -z "$$title" ]; then \
		echo "Error: Title is required"; \
		exit 1; \
	fi; \
	if [ -z "$$context" ]; then \
		gh workflow run adr-create.md -f title="$$title"; \
	else \
		gh workflow run adr-create.md -f title="$$title" -f context="$$context"; \
	fi; \
	echo ""; \
	echo "✅ ADR creation workflow triggered!"; \
	echo ""; \
	echo "The workflow will:"; \
	echo "  1. Generate the next ADR number"; \
	echo "  2. Create a comprehensive ADR document"; \
	echo "  3. Update the ADR index"; \
	echo "  4. Open a pull request for review"; \
	echo ""; \
	echo "Check workflow status: gh run list --workflow=adr-create.md"; \
	echo "View latest run: gh run view"

