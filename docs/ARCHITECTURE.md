# Architecture

## Engine mixin architecture

This section documents the internal design of `BasanosEngine` and the
`self: _EngineProtocol` convention that keeps the engine codebase
statically verifiable by type checkers (`ty`, `mypy`, `pyright`) while
preserving clean IDE navigation (Go-to-Definition, auto-complete).

### Overview

`BasanosEngine` is a **frozen dataclass** that inherits from three private
mixin classes, each defined in its own module:

| Mixin | Module | Responsibility |
|---|---|---|
| `_SolveMixin` | `_engine_solve.py` | `_iter_matrices`, `_iter_solve`, `warmup_state` |
| `_DiagnosticsMixin` | `_engine_diagnostics.py` | `condition_number`, `effective_rank`, `solver_residual`, `signal_utilisation` |
| `_SignalEvaluatorMixin` | `_engine_ic.py` | `ic`, `rank_ic`, IC summary statistics |

```python
@dataclasses.dataclass(frozen=True)
class BasanosEngine(_DiagnosticsMixin, _SignalEvaluatorMixin, _SolveMixin):
    prices: pl.DataFrame
    mu: pl.DataFrame
    cfg: BasanosConfig
```

Splitting the implementation across modules keeps each file focused and
independently testable, while `BasanosEngine` remains a thin facade that
wires them together.

### The `_EngineProtocol` contract

The private module `_engine_protocol.py` defines `_EngineProtocol`, a
[`typing.Protocol`](https://peps.python.org/pep-0544/) that enumerates the
attributes and methods that mixin implementations may access on `self`:

```python
class _EngineProtocol(Protocol):
    assets: list[str]
    prices: pl.DataFrame
    mu: pl.DataFrame
    cfg: BasanosConfig
    cor: dict[datetime.date, np.ndarray]
    ret_adj: pl.DataFrame
    vola: pl.DataFrame

    def _iter_matrices(self) -> ...: ...
    def _iter_solve(self) -> ...: ...
    def _ic_series(self, use_rank: bool) -> ...: ...
```

### The `self: _EngineProtocol` helper pattern

Every mixin method that accesses engine attributes must annotate its `self`
parameter with `_EngineProtocol`:

```python
# _engine_diagnostics.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._engine_protocol import _EngineProtocol


class _DiagnosticsMixin:
    @property
    def condition_number(self: _EngineProtocol) -> pl.DataFrame:
        # self.assets, self.prices, self._iter_matrices() are fully typed
        ...
```

**Why this works:**

1. The `TYPE_CHECKING` guard keeps the import out of the runtime critical path
   (avoiding circular imports).
2. At type-check time, `ty`/`mypy`/`pyright` see the full `_EngineProtocol`
   type for `self`, so every attribute access is validated.
3. At runtime, Python resolves `self` through normal MRO without ever needing
   the import.

**When to use plain `self` instead:**

Methods that only access attributes or properties defined *on the mixin
itself* (e.g. `ic_mean` calling `self.ic` where `ic` is a property on
`_SignalEvaluatorMixin`) should use plain `self`.  Adding
`self: _EngineProtocol` to such methods would be incorrect because
`_EngineProtocol` does not declare the mixin's own properties.

### Adding a new engine method

1. Decide which private module owns the implementation
   (`_engine_solve.py`, `_engine_diagnostics.py`, or `_engine_ic.py`).
2. Check whether any new attributes your method needs are already in
   `_EngineProtocol`.  If not, add them there first.
3. Write the method with `self: _EngineProtocol` (imported under
   `TYPE_CHECKING`) if it accesses engine attributes.
4. No changes to `optimizer.py` are needed — the method is automatically
   available on `BasanosEngine` via inheritance.
5. Run `make typecheck` to confirm zero type-check errors.

The tests in `tests/test_math/test_engine_protocol.py` enforce that every
method listed in the `_MUST_USE_PROTOCOL` table carries the correct `self`
annotation.  When you add a new method, add it to that table as well.

### Module dependency graph

```
optimizer.py (BasanosEngine)
    │
    ├── _engine_solve.py  (_SolveMixin)       ──► _engine_protocol.py
    ├── _engine_diagnostics.py (_DiagnosticsMixin) ──► _engine_protocol.py
    ├── _engine_ic.py (_SignalEvaluatorMixin)  ──► _engine_protocol.py
    │
    ├── _config.py (BasanosConfig, covariance configs)
    ├── _ewm_corr.py (ewm_corr, shared EWM math)
    ├── _factor_model.py (FactorModel)
    ├── _linalg.py (solve, inv_a_norm, valid)
    └── _signal.py (vol_adj, shrink2id)
```

`_engine_protocol.py` is imported **only** under `TYPE_CHECKING` in each
private module.  It is excluded from test coverage because its body consists
entirely of `Protocol` stubs (structural type annotations, not executable
code).

---

# Rhiza Architecture

Visual diagrams of Rhiza's architecture and component interactions.

## System Overview

```mermaid
flowchart TB
    subgraph User["User Interface"]
        make[make commands]
        local[local.mk]
    end

    subgraph Core[".rhiza/ Core"]
        rhizamk[rhiza.mk<br/>Core Logic]
        maked[make.d/*.mk<br/>Extensions]
        reqs[requirements/<br/>Dependencies]
        template[template-bundles.yml<br/>Bundle Config]
    end

    subgraph Config["Configuration"]
        pyproject[pyproject.toml]
        ruff[ruff.toml]
        precommit[.pre-commit-config.yaml]
        editorconfig[.editorconfig]
    end

    subgraph CI["GitHub Actions"]
        ci[CI Workflow]
        release[Release Workflow]
        security[Security Workflow]
        sync[Sync Workflow]
    end

    make --> rhizamk
    local -.-> rhizamk
    rhizamk --> maked
    maked --> reqs
    maked --> pyproject
    ci --> make
    release --> make
    security --> make
    sync --> template
```

## Makefile Hierarchy

```mermaid
flowchart TD
    subgraph Entry["Entry Point"]
        Makefile[Makefile<br/>9 lines]
    end

    subgraph Core["Core Logic"]
        rhizamk[.rhiza/rhiza.mk<br/>268 lines]
    end

    subgraph Extensions["Auto-loaded Extensions"]
        config[00-19: Configuration]
        tasks[20-79: Task Definitions]
        hooks[80-99: Hook Implementations]
    end

    subgraph Local["Local Customization"]
        localmk[local.mk<br/>Not synced]
    end

    Makefile -->|includes| rhizamk
    rhizamk -->|includes| config
    rhizamk -->|includes| tasks
    rhizamk -->|includes| hooks
    rhizamk -.->|optional| localmk
```

## Hook System

```mermaid
flowchart LR
    subgraph Hooks["Double-Colon Targets"]
        pre_install[pre-install::]
        post_install[post-install::]
        pre_sync[pre-sync::]
        post_sync[post-sync::]
        pre_release[pre-release::]
        post_release[post-release::]
        pre_bump[pre-bump::]
        post_bump[post-bump::]
    end

    subgraph Targets["Main Targets"]
        install[make install]
        sync[make sync]
        release[make release]
        publish[make publish]
        bump[make bump]
    end

    pre_install --> install --> post_install
    pre_sync --> sync --> post_sync
    pre_release --> release --> post_release
    pre_bump --> bump --> post_bump
```

## Release Pipeline

```mermaid
flowchart TD
    tag[Push Tag v*] --> validate[Validate Tag]
    validate --> build[Build Package]
    build --> draft[Draft GitHub Release]
    draft --> pypi[Publish to PyPI]
    draft --> devcontainer[Publish Devcontainer]
    pypi --> finalize[Finalize Release]
    devcontainer --> finalize

    subgraph Conditions
        pypi_cond{Has dist/ &<br/>not Private?}
        dev_cond{PUBLISH_DEVCONTAINER<br/>= true?}
    end

    draft --> pypi_cond
    pypi_cond -->|yes| pypi
    pypi_cond -->|no| finalize
    draft --> dev_cond
    dev_cond -->|yes| devcontainer
    dev_cond -->|no| finalize
```

## Template Sync Flow

```mermaid
flowchart LR
    upstream[Upstream Rhiza<br/>jebel-quant/rhiza] -->|template.yml| sync[make sync]
    sync -->|updates| downstream[Downstream Project]

    subgraph Synced["Synced Files"]
        workflows[.github/workflows/]
        rhiza[.rhiza/]
        configs[Config Files]
    end

    subgraph Preserved["Preserved"]
        localmk[local.mk]
        src[src/]
        tests[tests/]
    end

    sync --> Synced
    downstream --> Preserved
```

## Directory Structure

```mermaid
flowchart TD
    root[Project Root]

    root --> rhiza[.rhiza/]
    root --> github[.github/]
    root --> src[src/]
    root --> tests[tests/]
    root --> docs[docs/]
    root --> book[book/]

    rhiza --> rhizamk[rhiza.mk]
    rhiza --> maked[make.d/]
    rhiza --> reqs[requirements/]
    rhiza --> rtests[tests/]
    rhiza --> rdocs[docs/]
    rhiza --> templates[templates/]
    rhiza --> assets[assets/]

    github --> workflows[workflows/]
    workflows --> ci[rhiza_ci.yml]
    workflows --> release[rhiza_release.yml]
    workflows --> security[rhiza_security.yml]
    workflows --> more[... 11 more]

    maked --> agentic[agentic.mk]
    maked --> book[book.mk]
    maked --> bootstrap[bootstrap.mk]
    maked --> docker[docker.mk]
    maked --> docs_mk[docs.mk]
    maked --> github_mk[github.mk]
    maked --> marimo[marimo.mk]
    maked --> test[test.mk]
    maked --> more_mk[... 6 more]
```

## .rhiza/ Directory Structure and Dependencies

```mermaid
flowchart TB
    subgraph rhiza[".rhiza/ Directory"]
        direction TB
        
        subgraph core["Core Files"]
            rhizamk[rhiza.mk<br/>Core Logic - 153 lines]
            cfg[.cfg.toml<br/>Configuration]
            env[.env<br/>Environment]
            version[.rhiza-version<br/>Version]
            bundles[template-bundles.yml<br/>Bundle Definitions]
        end
        
        subgraph maked["make.d/ (14 files, ~41KB)"]
            direction LR
            agentic[agentic.mk<br/>AI Agents]
            bootstrap[bootstrap.mk<br/>Installation]
            test[test.mk<br/>Testing]
            book_mk[book.mk<br/>Documentation]
            docker_mk[docker.mk<br/>Containers]
            quality[quality.mk<br/>Code Quality]
            releasing[releasing.mk<br/>Releases]
            more[...]
        end
        
        subgraph requirements["requirements/ (4 files)"]
            direction LR
            tests_txt[tests.txt<br/>pytest, coverage]
            marimo_txt[marimo.txt<br/>notebooks]
            docs_txt[docs.txt<br/>pdoc]
            tools_txt[tools.txt<br/>pre-commit]
        end
        
        subgraph tests_dir["tests/ (23 files)"]
            direction LR
            api[api/<br/>Makefile Tests]
            integration[integration/<br/>E2E Tests]
            structure[structure/<br/>Layout Tests]
            sync[sync/<br/>Sync Tests]
            deps[deps/<br/>Dependency Tests]
        end
        
        subgraph other["Other Directories"]
            direction LR
            docs_dir[docs/<br/>7 MD files]
            assets_dir[assets/<br/>Logo]
        end
    end
    
    subgraph project["Project Files"]
        Makefile[Makefile<br/>Entry Point]
        pyproject[pyproject.toml<br/>Dependencies]
        ruff_toml[ruff.toml<br/>Linting]
        pytest_ini[pytest.ini<br/>Test Config]
        python_version[.python-version<br/>Python 3.13]
    end
    
    Makefile -->|includes| rhizamk
    rhizamk -->|auto-loads| maked
    maked -->|reads| pyproject
    maked -->|reads| python_version
    test -->|uses| pytest_ini
    test -->|installs| tests_txt
    book_mk -->|installs| docs_txt
    book_mk -->|uses| marimo_txt
    quality -->|uses| ruff_toml
    bootstrap -->|installs| tools_txt
    tests_dir -->|validates| core
    tests_dir -->|validates| maked
```

## CI/CD Workflow Triggers

```mermaid
flowchart TD
    subgraph Triggers
        push[Push]
        pr[Pull Request]
        schedule[Schedule]
        manual[Manual]
        tag[Tag v*]
    end

    subgraph Workflows
        ci[CI]
        security[Security]
        codeql[CodeQL]
        release[Release]
        deptry[Deptry]
        precommit[Pre-commit]
    end

    push --> ci
    push --> security
    push --> codeql
    pr --> ci
    pr --> deptry
    pr --> precommit
    schedule --> security
    manual --> ci
    tag --> release
```

## Python Execution Model

```mermaid
flowchart LR
    subgraph Commands
        make[make test]
        direct[Direct Python]
    end

    subgraph UV["uv Layer"]
        uv_run[uv run]
        uvx[uvx]
    end

    subgraph Tools
        pytest[pytest]
        ruff[ruff]
        hatch[hatch]
    end

    make --> uv_run
    uv_run --> pytest
    uv_run --> ruff
    uvx --> hatch

    direct -.->|Never| pytest

    style direct stroke-dasharray: 5 5
```

## Naming Conventions and Organization Patterns

### Makefile Naming (`.rhiza/make.d/`)

Makefiles follow these conventions:

1. **Lowercase with hyphens**: All makefile names use lowercase letters with hyphens for word separation
   - ✅ `bootstrap.mk`, `custom-task.mk`, `github.mk`
   - ❌ `Bootstrap.mk`, `customTask.mk`, `GitHub.mk`

2. **Descriptive domain names**: Each file represents a logical domain or feature area
   - `agentic.mk` - AI agent integrations
   - `bootstrap.mk` - Installation and setup
   - `docker.mk` - Docker containerization
   - `marimo.mk` - Marimo notebooks
   - `test.mk` - Testing infrastructure

3. **Example vs. Production files**:
   - Files prefixed with `custom-` are **examples** for user customization
   - `custom-env.mk` - Example environment variable customizations
   - `custom-task.mk` - Example custom task definitions
   - Users should create their own files or modify the root `Makefile` for customizations

### Target Naming

Make targets follow consistent patterns:

1. **Lowercase with hyphens**: Target names use lowercase with hyphens
   - ✅ `install-uv`, `docker-build`, `view-prs`
   - ❌ `installUv`, `docker_build`, `viewPRs`

2. **Verb-noun pattern**: Action-oriented targets use verb-noun format
   - `install-uv` - Install the uv tool
   - `docker-build` - Build Docker image
   - `view-prs` - View pull requests

3. **Namespace prefixes**: Related targets share a common prefix
   - Docker: `docker-build`, `docker-run`, `docker-clean`
   - LFS: `lfs-install`, `lfs-pull`, `lfs-track`, `lfs-status`
   - GitHub: `gh-install`, `view-prs`, `view-issues`, `failed-workflows`

### Section Headers (`##@`)

Section headers in makefiles group related targets in help output:

1. **Title Case**: Section names use Title Case
   - `##@ Bootstrap`
   - `##@ GitHub Helpers`
   - `##@ Marimo Notebooks`

2. **Descriptive grouping**: Sections group logically related commands
   - **Bootstrap** - Installation and setup
   - **Development and Testing** - Core dev workflow
   - **Documentation** - Doc generation
   - **GitHub Helpers** - GitHub CLI integrations
   - **Quality and Formatting** - Code quality tools

### Hook Naming

Hook targets use double-colon syntax and follow a `pre-`/`post-` pattern:

```makefile
pre-install::    # Runs before make install
post-install::   # Runs after make install
pre-sync::       # Runs before make sync
post-sync::      # Runs after make sync
pre-release::    # Runs before make release
post-release::   # Runs after make release
```

**Key principles**:
- Always use double-colon (`::`) to allow multiple definitions
- Hooks are defined as phony targets
- Empty default implementations use `; @:` syntax

### File Organization Patterns

1. **Directory naming**:
   - Lowercase with hyphens: `make.d/`, `template-bundles.yml`
   - Plural for collections: `requirements/`, `templates/`, `tests/`

2. **Test organization** (`.rhiza/tests/`):
   - Tests grouped by **purpose**, not by feature
   - `api/` - Makefile API tests
   - `structure/` - Project structure validation
   - `integration/` - End-to-end workflows
   - `sync/` - Template synchronization
   - `deps/` - Dependency validation

3. **Requirements organization** (`.rhiza/requirements/`):
   - Named by **purpose**: `tests.txt`, `docs.txt`, `marimo.txt`, `tools.txt`
   - Not by library: ❌ `pytest.txt`, `pdoc.txt`

### Template Bundle Naming

Template bundles in `template-bundles.yml` follow these conventions:

1. **Lowercase singular**: `core`, `github`, `tests`, `marimo`, `book`
2. **Domain-focused**: Named after the feature domain, not implementation
   - ✅ `marimo` (notebooks)
   - ✅ `book` (documentation)
   - ❌ `notebooks`, `documentation-generation`

3. **Bundle metadata**:
   - `description` - Clear, concise explanation
   - `standalone` - Whether bundle can be used independently
   - `requires` - Hard dependencies on other bundles
   - `recommends` - Soft dependencies that enhance functionality

### Variable Naming

Makefile variables follow these patterns:

1. **SCREAMING_SNAKE_CASE**: All uppercase with underscores
   - `INSTALL_DIR`, `UV_BIN`, `PYTHON_VERSION`, `VENV`

2. **Suffix patterns**:
   - `_BIN` - Executable paths: `UV_BIN`, `UVX_BIN`, `COPILOT_BIN`
   - `_DIR` - Directory paths: `INSTALL_DIR`, `DOCKER_FOLDER`
   - `_VERSION` - Version strings: `PYTHON_VERSION`, `RHIZA_VERSION`

3. **Namespace prefixes**: Related variables share prefixes
   - UV tooling: `UV_BIN`, `UVX_BIN`, `UV_LINK_MODE`
   - Color codes: `BLUE`, `GREEN`, `RED`, `YELLOW`, `RESET`, `BOLD`

### Documentation Naming

Documentation files use SCREAMING_SNAKE_CASE:

- `README.md` - Directory/project overview
- `ARCHITECTURE.md` - Architecture diagrams
- `CUSTOMIZATION.md` - Customization guide
- `QUICK_REFERENCE.md` - Command reference
- `SECURITY.md` - Security policy

### Workflow Naming (`.github/workflows/`)

GitHub Actions workflows use the pattern `rhiza_<feature>.yml`:

- `rhiza_ci.yml` - Continuous integration
- `rhiza_release.yml` - Release automation
- `rhiza_security.yml` - Security scanning
- `rhiza_sync.yml` - Template synchronization
- `rhiza_deptry.yml` - Dependency checking

**Rationale**: The `rhiza_` prefix clearly identifies template-managed workflows, distinguishing them from user-defined workflows.

## Key Design Principles

### 1. Single Source of Truth

- **Python version**: `.python-version` file (not hardcoded)
- **Rhiza version**: `.rhiza/.rhiza-version` file
- **Dependencies**: `pyproject.toml` (not duplicated in makefiles)
- **Bundle definitions**: `template-bundles.yml` (not scattered)

### 2. Auto-Loading Pattern

Makefiles in `.rhiza/make.d/` are automatically included:

```makefile
# In .rhiza/rhiza.mk (last line)
-include .rhiza/make.d/*.mk
```

This allows:
- Adding new features by dropping in a `.mk` file
- No manual maintenance of include lists
- Clean separation of concerns

### 3. Extension Points

Users can extend Rhiza without modifying template files:

1. **Root Makefile**: Add custom targets before `include .rhiza/rhiza.mk`
2. **local.mk**: Local shortcuts (not committed, auto-loaded)
3. **Hooks**: Use double-colon targets (`post-install::`, etc.)

### 4. Fail-Safe Defaults

- Missing tools are detected and installation offered
- Missing directories are created automatically
- Graceful degradation when optional features are unavailable

### 5. Documentation as Code

- Every target has a `##` help comment
- Section headers (`##@`) organize help output
- README files in every major directory
- Comprehensive INDEX.md for quick reference
