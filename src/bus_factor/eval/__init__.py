"""Evaluation — the part that turns 'I built a RAG app' into a defensible claim."""

from bus_factor.eval.dataset import load_qa_pairs, split_holdout
from bus_factor.eval.harness import EvalReport, make_leave_one_out_factory, run_eval

__all__ = [
    "EvalReport",
    "load_qa_pairs",
    "make_leave_one_out_factory",
    "run_eval",
    "split_holdout",
]
