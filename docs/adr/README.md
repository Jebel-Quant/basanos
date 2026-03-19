# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the Basanos project.

An ADR captures an important architectural decision made along with its context and consequences.

## Index

| # | Title | Status | Date |
|---|-------|--------|------|
| [0001](0001-rhiza-framework-dependency.md) | Rhiza Framework Dependency | Accepted | 2026-03-19 |

## Format

Each ADR follows this structure:

- **Title** — Short noun phrase naming the decision
- **Status** — Proposed / Accepted / Deprecated / Superseded
- **Context** — The problem and forces at play
- **Decision** — The chosen response to those forces
- **Consequences** — What becomes easier or harder as a result
- **Alternatives Considered** — Other options that were evaluated

## Adding a New ADR

Run `make adr` from the project root. The workflow will generate the next ADR number, draft the document, and open a pull request for review.
