"""Run the answerer over a holdout set and produce a defensible report.

The report deliberately carries more than an accuracy number:

* accuracy / mean-F1 / mean-ROUGE-L  -> did it reproduce the human's answer?
* retrieval-hit-rate                 -> did the *right* source even get retrieved?
                                        (separates retrieval failures from
                                        generation failures — you cannot fix
                                        what you cannot attribute)
* ECE + a calibration table          -> is confidence meaningful?
* abstention-rate                    -> how often it correctly says 'I don't know'

Every number is stamped with which judge and which provider produced it, because
a metric without that context is theatre.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from bus_factor.agent.answerer import Answerer
from bus_factor.config import Settings
from bus_factor.eval.metrics import (
    expected_calibration_error,
    mean,
    rouge_l,
    token_f1,
)
from bus_factor.memory.store import MemoryStore
from bus_factor.models import Document, QAPair

_ABSTENTION_MARKERS = ("i don't have", "i don't know", "not enough recorded")


def _is_abstention(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _ABSTENTION_MARKERS)


@dataclass
class EvalReport:
    summary: dict[str, Any]
    items: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "items": self.items}

    def to_pretty(self) -> str:
        s = self.summary
        lines = [
            "",
            "=" * 60,
            "  BUS FACTOR — evaluation report",
            "=" * 60,
            f"  holdout size        : {s['n']}",
            f"  provider            : {s['provider']}",
            f"  judge               : {s['judge']}",
            "-" * 60,
            f"  accuracy (judge)    : {s['accuracy']:.1%}",
            f"  mean token-F1       : {s['mean_f1']:.3f}",
            f"  mean ROUGE-L        : {s['mean_rouge_l']:.3f}",
            f"  retrieval hit-rate  : {s['retrieval_hit_rate']:.1%}",
            f"  abstention rate     : {s['abstention_rate']:.1%}",
            "-" * 60,
            f"  mean confidence     : {s['mean_confidence']:.3f}",
            f"  calibration ECE     : {s['ece']:.3f}   (0 = perfect)",
            "  accuracy by confidence bucket:",
        ]
        for label, bucket in s["calibration_table"].items():
            if bucket["n"]:
                lines.append(
                    f"    {label:<14} n={bucket['n']:<3} acc={bucket['accuracy']:.1%}"
                )
        lines += ["=" * 60, ""]
        return "\n".join(lines)


def belongs_to_source(doc: Document, source_id: str) -> bool:
    """True if a document originates from the given issue/source."""
    return doc.source.id == source_id or doc.source.id.startswith(source_id + "-")


def make_leave_one_out_factory(
    documents: list[Document], settings: Settings, use_dense: bool = False
) -> Callable[[QAPair], Answerer]:
    """Build a per-question answerer whose store excludes that question's own answer.

    This is what turns the eval from 'can it echo the stored answer' (trivial)
    into 'can it answer from the person's *other* recorded knowledge' (the real
    generalization question, and the one that stresses the trust layer).
    """

    def factory(qa: QAPair) -> Answerer:
        kept = [d for d in documents if not belongs_to_source(d, qa.source.id)]
        store = MemoryStore(rrf_k=settings.rrf_k, use_dense=use_dense)
        store.add(kept)
        return Answerer(store, settings)

    return factory


def run_eval(
    holdout: list[QAPair],
    judge,
    *,
    answerer: Answerer | None = None,
    answerer_factory: Callable[[QAPair], Answerer] | None = None,
    as_of: date | None = None,
) -> EvalReport:
    """Score the answerer over a holdout set.

    Pass ``answerer`` for closed-book evaluation over a fixed store, or
    ``answerer_factory`` to construct a fresh answerer per question (used for
    leave-one-out).
    """
    if answerer is None and answerer_factory is None:
        raise ValueError("provide either answerer or answerer_factory")

    items: list[dict[str, Any]] = []
    confidences: list[float] = []
    correct_flags: list[bool] = []
    provider_name = answerer.provider.name if answerer else "unknown"

    for qa in holdout:
        active = answerer_factory(qa) if answerer_factory else answerer
        provider_name = active.provider.name
        ans = active.answer(qa.question, as_of=as_of)
        verdict = judge.judge(qa.question, qa.reference_answer, ans.text)
        f1 = token_f1(ans.text, qa.reference_answer)
        rouge = rouge_l(ans.text, qa.reference_answer)
        retrieval_hit = any(
            c.document_id == qa.source.id or c.document_id.startswith(qa.source.id + "-")
            for c in ans.citations
        )
        items.append(
            {
                "id": qa.id,
                "question": qa.question[:200],
                "reference": qa.reference_answer[:300],
                "prediction": ans.text[:300],
                "correct": verdict.correct,
                "judge_score": verdict.score,
                "f1": round(f1, 4),
                "rouge_l": round(rouge, 4),
                "confidence": ans.confidence,
                "retrieval_hit": retrieval_hit,
                "staleness_days": ans.staleness_days,
                "abstained": _is_abstention(ans.text),
                "n_evidence": ans.meta.get("n_evidence", 0),
            }
        )
        confidences.append(ans.confidence)
        correct_flags.append(verdict.correct)

    n = len(holdout)
    calibration = _calibration_table(confidences, correct_flags)

    summary = {
        "n": n,
        "provider": provider_name,
        "judge": getattr(judge, "name", type(judge).__name__),
        "accuracy": mean([1.0 if c else 0.0 for c in correct_flags]),
        "mean_f1": mean([i["f1"] for i in items]),
        "mean_rouge_l": mean([i["rouge_l"] for i in items]),
        "retrieval_hit_rate": mean([1.0 if i["retrieval_hit"] else 0.0 for i in items]),
        "abstention_rate": mean([1.0 if i["abstained"] else 0.0 for i in items]),
        "mean_confidence": mean(confidences),
        "ece": expected_calibration_error(confidences, correct_flags),
        "calibration_table": calibration,
    }
    return EvalReport(summary=summary, items=items)


def _calibration_table(
    confidences: list[float], correct: list[bool]
) -> dict[str, dict[str, Any]]:
    """Accuracy grouped into low/medium/high confidence bands.

    A healthy system shows monotonically rising accuracy across the bands — that
    is the visual proof that confidence carries signal.
    """
    bands = {
        "low[0,.35)": (0.0, 0.35),
        "med[.35,.6)": (0.35, 0.6),
        "high[.6,1]": (0.6, 1.01),
    }
    table: dict[str, dict[str, Any]] = {}
    for label, (lo, hi) in bands.items():
        idx = [i for i, c in enumerate(confidences) if lo <= c < hi]
        acc = mean([1.0 if correct[i] else 0.0 for i in idx]) if idx else 0.0
        table[label] = {"n": len(idx), "accuracy": acc}
    return table
