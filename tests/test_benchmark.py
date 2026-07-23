"""Guards the synthetic benchmark's key properties (and thus the calibration story)."""

from __future__ import annotations

from bus_factor.benchmark import generate_corpus, load_benchmark, run_benchmark


def test_corpus_is_deterministic():
    assert generate_corpus() == generate_corpus()


def test_corpus_has_documents_and_qa():
    docs, qa = load_benchmark()
    assert len(docs) > 50
    assert len(qa) > 50


def test_benchmark_is_nondegenerate():
    r = run_benchmark()
    closed = r["closed_book"].summary
    raw = r["leave_one_out"].summary
    cal = r["leave_one_out_calibrated"].summary

    # Captured knowledge is faithfully recalled when present.
    assert closed["accuracy"] >= 0.9
    # Leave-one-out is a genuine mix (recoverable clusters + unrecoverable
    # singletons), not the degenerate all-or-nothing of the tiny demo fixture.
    assert 0.3 < raw["accuracy"] < 0.9
    # Calibration triggers real abstention...
    assert cal["coverage"] < 1.0
    # ...that makes the answers it *does* give more trustworthy than answering all,
    assert cal["selective_accuracy"] > raw["accuracy"]
    # and it does not worsen calibration error.
    assert cal["ece"] <= raw["ece"]
