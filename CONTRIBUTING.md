# Contributing

Thanks for looking. This is a portfolio project, but it is held to a real bar.

## Setup

```bash
make setup        # editable install with dev + fast-retrieval extras
```

## The gate

Everything must pass before a change lands. One command runs all of it:

```bash
make check        # ruff (lint) + mypy (types) + pytest with coverage
```

Individually:

```bash
make lint         # ruff check src tests
make typecheck    # mypy, must be clean
make coverage     # pytest with coverage; fails under 80%
make demo         # end-to-end smoke test, offline
```

CI runs the same gate on Python 3.11 and 3.12.

## Principles

- **The core stays dependency-free.** New third-party dependencies go behind an
  optional extra in `pyproject.toml` and must degrade gracefully when absent.
- **Separate I/O from logic.** Network/disk calls live behind thin functions;
  the logic they feed is pure and unit-tested without them.
- **Every answer keeps its provenance.** Do not add a path that returns text
  without citations, confidence, and staleness.
- **Numbers must be reproducible.** Anything affecting a metric must be
  deterministic (no unseeded randomness, no wall-clock in the hot path).
- **Type everything.** New public functions carry annotations; `mypy` stays green.

## Tests

Add tests next to the behaviour you change (`tests/`). Keep them offline — no
test may require network or an API key. Use fixtures for anything I/O-shaped.
