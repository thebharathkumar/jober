# Bus Factor — developer entry points.
# The core needs no install; targets set PYTHONPATH=src so everything runs from
# a fresh clone. `make setup` is only needed for the optional dev tooling.

export PYTHONPATH := src
PY ?= python3

.PHONY: help setup lint format test demo eval clean

help:
	@echo "make demo     - run the full pipeline offline and print the eval report"
	@echo "make test     - run the test suite (no network)"
	@echo "make lint     - ruff check"
	@echo "make format   - ruff format"
	@echo "make setup    - install dev + optional dependencies"
	@echo "make clean    - remove caches, reports, and generated data"

setup:
	$(PY) -m pip install -e ".[dev,fast-retrieval]"

lint:
	$(PY) -m ruff check src tests

format:
	$(PY) -m ruff format src tests

test:
	$(PY) -m pytest

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
