# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Synthetic benchmark** (`bus_factor/benchmark.py`, `scripts/synthetic_benchmark.py`):
  a deterministic ~100-doc corpus mixing recoverable clusters with unrecoverable
  singletons, so calibration is non-degenerate. Shows leave-one-out selective
  accuracy 60% → 87% and ECE 0.16 → 0.07 via abstention.
- Extractive provider now answers from the highest-ranked retrieved document
  (trust-the-retriever) instead of re-competing sentences across all of them,
  fixing cases where the right source was retrieved but a neighbour's sentence
  was returned.
- **Confidence calibration**: `IsotonicCalibrator` (hand-rolled weighted PAVA,
  zero-dep) maps raw retrieval confidence to empirical correctness, fit on a
  held-out split via `BusFactor.fit_calibration`. The answerer abstains below a
  calibrated `abstain_threshold`. Leave-one-out ECE drops 0.77 → ~0.00.
- **Selective metrics** in the eval report: `coverage` and `selective_accuracy`
  (accuracy when the system chooses to answer), plus a third demo mode showing
  the before/after calibration contrast.
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
