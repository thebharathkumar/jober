"""The public library API.

Everything the CLI does is available programmatically through one object, so
Bus Factor can be embedded in a service or notebook without shelling out:

    from bus_factor import BusFactor
    bf = BusFactor.from_sqlite("data/knowledge.db")
    answer = bf.ask("How do I rotate the signing key?")
    report = bf.evaluate(qa_pairs, leave_one_out=True)

Construction is separated from data loading via classmethods so the source of
truth (SQLite, JSONL, or in-memory documents) is a caller choice, not baked in.
"""

from __future__ import annotations

from pathlib import Path

from bus_factor.agent.answerer import Answerer
from bus_factor.calibration import IsotonicCalibrator
from bus_factor.config import Settings
from bus_factor.eval.dataset import load_documents, split_holdout
from bus_factor.eval.harness import (
    EvalReport,
    fit_calibrator,
    make_leave_one_out_factory,
    run_eval,
)
from bus_factor.eval.judge import get_judge
from bus_factor.logging_config import get_logger
from bus_factor.memory.persistence import SqliteKnowledgeBase
from bus_factor.memory.store import MemoryStore
from bus_factor.models import Answer, Document, QAPair

log = get_logger(__name__)


class BusFactor:
    """A configured knowledge-continuity engine: store + answerer + evaluation."""

    def __init__(self, settings: Settings | None = None, use_dense: bool = False) -> None:
        self.settings = settings or Settings.from_env()
        self.store = MemoryStore(rrf_k=self.settings.rrf_k, use_dense=use_dense)
        self.answerer = Answerer(self.store, self.settings)

    # --- construction -------------------------------------------------------

    def add_documents(self, docs: list[Document]) -> BusFactor:
        self.store.add(docs)
        log.info("indexed %d documents (%d total)", len(docs), len(self.store.documents))
        return self

    @classmethod
    def from_documents(
        cls, docs: list[Document], settings: Settings | None = None, use_dense: bool = False
    ) -> BusFactor:
        return cls(settings=settings, use_dense=use_dense).add_documents(docs)

    @classmethod
    def from_jsonl(
        cls, docs_path: str | Path, settings: Settings | None = None, use_dense: bool = False
    ) -> BusFactor:
        return cls.from_documents(load_documents(docs_path), settings, use_dense)

    @classmethod
    def from_sqlite(
        cls, db_path: str | Path, settings: Settings | None = None, use_dense: bool = False
    ) -> BusFactor:
        with SqliteKnowledgeBase(db_path) as kb:
            docs = kb.documents()
        return cls.from_documents(docs, settings, use_dense)

    # --- use ----------------------------------------------------------------

    def ask(self, question: str) -> Answer:
        return self.answerer.answer(question)

    def evaluate(
        self, qa_pairs: list[QAPair], leave_one_out: bool = False, judge: object | None = None
    ) -> EvalReport:
        judge = judge or get_judge(self.settings)
        _, holdout = split_holdout(qa_pairs, self.settings.holdout_fraction)
        if leave_one_out:
            factory = make_leave_one_out_factory(
                self.store.documents, self.settings, calibrator=self.answerer.calibrator
            )
            return run_eval(holdout, judge, answerer_factory=factory)
        return run_eval(holdout, judge, answerer=self.answerer)

    def fit_calibration(
        self, qa_pairs: list[QAPair], leave_one_out: bool = True
    ) -> IsotonicCalibrator:
        """Fit and install a confidence calibrator, then enable abstention.

        Fits on the TRAIN split (disjoint from the holdout ``evaluate`` scores on),
        in the same regime it will be applied to. After this, ``ask`` and
        ``evaluate`` return calibrated confidences and abstain below the threshold.
        """
        train, _ = split_holdout(qa_pairs, self.settings.holdout_fraction)
        judge = get_judge(self.settings)
        # Fit against an UNCALIBRATED answerer so the raw scores are learned cleanly.
        raw_answerer = Answerer(self.store, self.settings)
        if leave_one_out:
            factory = make_leave_one_out_factory(self.store.documents, self.settings)
            calibrator = fit_calibrator(train, judge, answerer_factory=factory)
        else:
            calibrator = fit_calibrator(train, judge, answerer=raw_answerer)
        self.answerer.calibrator = calibrator
        log.info("fitted confidence calibrator on %d calibration questions", len(train))
        return calibrator

    @property
    def document_count(self) -> int:
        return len(self.store.documents)
