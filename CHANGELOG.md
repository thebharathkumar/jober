# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Persistence**: `SqliteKnowledgeBase`, a durable single-file knowledge base
  with incremental (upsert-by-id) ingest. New `--db` option on `ingest`, `ask`,
  and `eval`, and a `stats` command.
- **Library API**: the `BusFactor` facade (`from_documents` / `from_jsonl` /
  `from_sqlite`, `ask`, `evaluate`) exported from the package root.
- **Typed error hierarchy** (`errors.py`) with central CLI error handling
  (clean message + exit code, no traceback).
- **Structured, library-safe logging** (`logging_config.py`), `--verbose` flag,
  and `BF_LOG_LEVEL` env var.
- **Configuration validation** (`Settings.__post_init__` → `ConfigError`).
- **Schema validation** on data load; malformed JSONL rows are skipped and logged.
- **Resilient GitHub client**: retry with exponential backoff, rate-limit
  detection (`Retry-After` / `X-RateLimit-Reset`), `404 → RepoNotFound`, and
  repo-slug validation.
- `py.typed` marker; `mypy`-clean.
- CI now runs type-checking and coverage (threshold-gated); coverage ~82%.
- `CONTRIBUTING.md`, `SECURITY.md`, this changelog.

## [0.1.0] — initial

### Added
- Four-layer architecture: `ingest` (closed issues → knowledge + eval set),
  `memory` (dependency-free BM25 + hybrid RRF store), `agent` (grounded answerer
  with citations/confidence/staleness; Anthropic + offline extractive provider),
  and `eval` (closed-book and leave-one-out harness with accuracy, F1, ROUGE-L,
  retrieval-hit-rate, abstention, and ECE calibration).
- `bus-factor` CLI: `demo`, `ingest`, `ask`, `eval`.
- Fully synthetic bundled sample corpus; offline `make demo`.
- Docs: README, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `ETHICS.md`.
