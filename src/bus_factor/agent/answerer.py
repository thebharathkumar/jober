"""Grounded answerer: retrieve -> answer -> attach provenance.

The contract this class enforces is the product's whole promise: an answer is
never returned as bare text. It always carries (a) citations to the exact
sources it drew from, (b) a confidence derived from retrieval quality, and
(c) the staleness of its freshest evidence. A caller can therefore make a
trust decision — accept, show-with-caveat, or route to a human — without
re-reading the corpus.

Confidence here is a *heuristic proxy* from retrieval scores. We do not claim
it is calibrated by construction; the eval harness measures calibration (ECE)
so the claim is earned, not asserted.
"""

from __future__ import annotations

from datetime import date

from bus_factor.agent.provider import LLMProvider, get_provider
from bus_factor.calibration import IsotonicCalibrator
from bus_factor.config import Settings
from bus_factor.memory.store import MemoryStore
from bus_factor.models import Answer, Citation

ABSTENTION_TEXT = "I don't have reliable recorded knowledge to answer that confidently."

PERSONA_SYSTEM = (
    "You are the recorded professional knowledge of a specific person, captured "
    "so their judgment survives after they leave a team. Answer questions the way "
    "the evidence shows they would, grounded strictly in the recorded knowledge "
    "provided. Prefer 'I don't know' over guessing. Every claim must trace to the "
    "evidence."
)


def _confidence_from_results(top_lexical: float, n_results: int) -> float:
    """Squash the top lexical score into [0, 1] as a retrieval-quality proxy.

    Saturating form ``x / (x + c)`` avoids an arbitrary cap and is monotonic in
    match strength. A small bonus for having multiple supporting hits reflects
    that corroborated retrieval is more trustworthy than a single lucky match.
    """
    base = top_lexical / (top_lexical + 6.0)
    corroboration = min(0.15, 0.05 * max(0, n_results - 1))
    return round(min(1.0, base + corroboration), 3)


class Answerer:
    def __init__(
        self,
        store: MemoryStore,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
        calibrator: IsotonicCalibrator | None = None,
    ) -> None:
        self.store = store
        self.settings = settings or Settings.from_env()
        self.provider = provider or get_provider(self.settings)
        # When set, raw retrieval confidence is mapped through the calibrator and
        # the answerer abstains below the (now trustworthy) abstain threshold.
        self.calibrator = calibrator

    def answer(self, question: str, as_of: date | None = None) -> Answer:
        as_of = as_of or date.today()
        results = self.store.search(question, k=self.settings.top_k, as_of=as_of)

        if not results:
            return Answer(
                text="I don't have any recorded knowledge relevant to that question.",
                confidence=0.0,
                staleness_days=None,
                meta={"provider": self.provider.name, "n_evidence": 0},
            )

        contexts = [self._format_context(r.document) for r in results]
        text = self.provider.complete(PERSONA_SYSTEM, question, contexts)

        citations = [
            Citation(
                document_id=r.document.id,
                title=r.document.source.title,
                url=r.document.source.url,
                score=round(r.score, 6),
                age_days=r.document.age_days(as_of),
            )
            for r in results
        ]
        ages = [c.age_days for c in citations if c.age_days is not None]
        staleness = min(ages) if ages else None  # freshest evidence drives the flag

        raw_confidence = _confidence_from_results(results[0].lexical_score, len(results))
        confidence = (
            round(self.calibrator.predict(raw_confidence), 3)
            if self.calibrator is not None
            else raw_confidence
        )

        # Abstain only when confidence is trustworthy (i.e. calibrated). Abstaining
        # on the raw, uncalibrated score would just move the overconfidence problem
        # around; once calibrated, a low number genuinely means "I don't know".
        abstained = False
        if self.calibrator is not None and confidence < self.settings.abstain_threshold:
            text = ABSTENTION_TEXT
            abstained = True

        meta = {
            "provider": self.provider.name,
            "n_evidence": len(results),
            "raw_confidence": raw_confidence,
            "calibrated": self.calibrator is not None,
            "abstained": abstained,
            "low_confidence": confidence < self.settings.low_confidence_threshold,
            "stale": staleness is not None and staleness > self.settings.staleness_warn_days,
        }
        return Answer(
            text=text,
            citations=citations,
            confidence=confidence,
            staleness_days=staleness,
            meta=meta,
        )

    @staticmethod
    def _format_context(doc) -> str:
        # Feed the authored body to the answerer, not the title. Titles are
        # derived from the *question* ("Re: how do I ..."), so including them
        # would tempt an extractive answerer into echoing the prompt. Retrieval
        # still uses the title for ranking; generation should not.
        return doc.text or doc.source.title
