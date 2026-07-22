"""Judges decide whether a predicted answer agrees with the reference.

Two judges ship, same interface:

* ``LexicalJudge`` — offline, deterministic; calls an answer correct when its
  token-F1 against the reference clears a threshold. Cheap and reproducible;
  the default so CI needs nothing.
* ``LLMJudge`` — semantic grading via a Claude model, used when a key is set.
  Catches correct-but-differently-worded answers the lexical judge misses.

Reporting which judge produced a number is mandatory — an accuracy figure is
meaningless without saying who graded it.
"""

from __future__ import annotations

from dataclasses import dataclass

from bus_factor.config import Settings
from bus_factor.eval.metrics import token_f1


@dataclass
class Verdict:
    correct: bool
    score: float  # 0..1
    rationale: str


class LexicalJudge:
    name = "lexical-f1"

    def __init__(self, threshold: float = 0.4) -> None:
        self.threshold = threshold

    def judge(self, question: str, reference: str, prediction: str) -> Verdict:
        f1 = token_f1(prediction, reference)
        return Verdict(
            correct=f1 >= self.threshold,
            score=round(f1, 4),
            rationale=f"token-F1={f1:.3f} vs threshold={self.threshold}",
        )


class LLMJudge:
    name = "llm-semantic"

    def __init__(self, settings: Settings) -> None:
        import anthropic  # noqa: PLC0415

        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.model

    def judge(self, question: str, reference: str, prediction: str) -> Verdict:
        prompt = (
            "You are grading whether a candidate answer conveys the same "
            "substantive guidance as the reference answer to a question.\n\n"
            f"Question: {question}\n\nReference answer: {reference}\n\n"
            f"Candidate answer: {prediction}\n\n"
            "Reply with exactly one line: a score 0.0-1.0 for semantic agreement, "
            "then a space, then YES if it is substantively correct or NO if not."
        )
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        try:
            score_str, verdict = raw.split()[0], raw.split()[-1]
            score = max(0.0, min(1.0, float(score_str)))
        except (ValueError, IndexError):
            score, verdict = 0.0, "NO"
        return Verdict(correct=verdict.upper().startswith("Y"), score=score, rationale=raw)


def get_judge(settings: Settings):
    """LLM judge when credentials exist, otherwise the deterministic lexical one."""
    if settings.has_llm:
        try:
            return LLMJudge(settings)
        except Exception:
            pass
    return LexicalJudge(threshold=settings.judge_f1_threshold)
