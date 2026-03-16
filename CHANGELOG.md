# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.3] - 2026-03-16

### Fixed

- `Stats.sharpe()` now returns `float('nan')` instead of raising `ZeroDivisionError`
  when the series has zero volatility (constant returns).

### Added

- `py.typed` marker added to the package so that PEP 561 type-checking discovery
  works correctly for downstream consumers.

## [0.2.2] - 2025-01-01

- Initial tracked release.
