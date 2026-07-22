# Bus Factor — developer entry points.
# The core needs no install; targets set PYTHONPATH=src so everything runs from
# a fresh clone. `make setup` is only needed for the optional dev tooling.

export PYTHONPATH := src
PY ?= python3

.PHONY: help setup lint typecheck test coverage check demo eval clean

help:
	@echo "make demo      - run the full pipeline offline and print the eval report"
	@echo "make test      - run the test suite (no network)"
	@echo "make coverage  - run tests with coverage (fails under threshold)"
	@echo "make lint      - ruff check"
	@echo "make typecheck - mypy"
	@echo "make check     - lint + typecheck + coverage (the full gate)"
	@echo "make setup     - install dev + optional dependencies"
	@echo "make clean     - remove caches, reports, and generated data"

setup:
	$(PY) -m pip install -e ".[dev,fast-retrieval]"

lint:
	$(PY) -m ruff check src tests

format:
	$(PY) -m ruff format src tests

typecheck:
	MYPYPATH=src $(PY) -m mypy src/bus_factor

test:
	$(PY) -m pytest

coverage:
	$(PY) -m pytest --cov=bus_factor --cov-report=term-missing

check: lint typecheck coverage

demo:
	$(PY) -m bus_factor.cli demo

# Requires a prior `bus-factor ingest <repo> --out data`.
eval:
	$(PY) -m bus_factor.cli eval --docs data/docs.jsonl --eval data/eval.jsonl
	$(PY) -m bus_factor.cli eval --docs data/docs.jsonl --eval data/eval.jsonl --leave-one-out

clean:
	rm -rf reports data .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
