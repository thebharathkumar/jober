"""Command-line entry point.

    bus-factor demo                      end-to-end on bundled sample data (offline)
    bus-factor ingest <owner/repo>       scrape a public repo -> docs + eval set
    bus-factor ask   "<question>"        answer from a knowledge base, with citations
    bus-factor eval                      score the answerer against a holdout set
    bus-factor stats --db <path>         summarize a persisted knowledge base

Built on argparse so the CLI itself adds zero dependencies. Errors raised by the
library are caught centrally and reported without a stack trace.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

from bus_factor import __version__
from bus_factor.app import BusFactor
from bus_factor.config import Settings
from bus_factor.errors import BusFactorError
from bus_factor.eval.dataset import load_qa_pairs, split_holdout
from bus_factor.ingest.github_issues import ingest_repo, transform_repo
from bus_factor.logging_config import configure_logging
from bus_factor.memory.persistence import SqliteKnowledgeBase
from bus_factor.models import Answer, Document, QAPair, write_jsonl


def _load_sample_pipeline() -> tuple[list[Document], list[QAPair]]:
    raw = json.loads(
        resources.files("bus_factor").joinpath("_data/sample_issues.json").read_text("utf-8")
    )
    pairs = [(entry["issue"], entry.get("comments", [])) for entry in raw]
    return transform_repo(pairs)


def _print_answer(question: str, ans: Answer) -> None:
    print(f"\nQ: {question}\n")
    print(f"A: {ans.text}\n")
    flags = []
    if ans.meta.get("low_confidence"):
        flags.append("LOW-CONFIDENCE")
    if ans.meta.get("stale"):
        flags.append("STALE")
    flag_str = f"  [{', '.join(flags)}]" if flags else ""
    stale = f"{ans.staleness_days:.0f}d" if ans.staleness_days is not None else "n/a"
    print(f"   confidence={ans.confidence:.2f}  freshest-evidence={stale}{flag_str}")
    if ans.citations:
        print("   sources:")
        for c in ans.citations[:5]:
            age = f"{c.age_days:.0f}d" if c.age_days is not None else "?"
            print(f"     - {c.title}  ({age} old)  {c.url}".rstrip())


def _engine_from_args(args: argparse.Namespace, settings: Settings) -> BusFactor:
    """Build an engine from either a SQLite knowledge base or a docs JSONL."""
    if getattr(args, "db", None):
        return BusFactor.from_sqlite(args.db, settings=settings, use_dense=args.dense)
    return BusFactor.from_jsonl(args.docs, settings=settings, use_dense=args.dense)


def _write_report(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nFull report written to {out}")


def cmd_demo(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    docs, qa = _load_sample_pipeline()
    bf = BusFactor.from_documents(docs, settings=settings, use_dense=args.dense)

    print(
        f"Loaded {len(docs)} knowledge documents and {len(qa)} ground-truth QA pairs "
        f"from bundled sample data."
    )

    # Mode 1 — closed-book: the answer IS in the store. Measures retrieval +
    # faithful reproduction of captured knowledge (can it recall what it learned).
    closed = bf.evaluate(qa)
    print("\n### Mode 1: closed-book (captured knowledge is retrievable)")
    print(closed.to_pretty())

    # Mode 2 — leave-one-out: each question's own answer is withheld, so the
    # system must answer from the person's OTHER knowledge. Measures generalization
    # and, crucially, whether the trust layer backs off when it truly doesn't know.
    loo = bf.evaluate(qa, leave_one_out=True)
    print("### Mode 2: leave-one-out, RAW confidence (uncalibrated)")
    print(loo.to_pretty())

    # Mode 3 — fit an isotonic calibrator on a held-out split, then re-run LOO.
    # Confidence becomes trustworthy and the system abstains instead of guessing.
    bf.fit_calibration(qa, leave_one_out=True)
    loo_cal = bf.evaluate(qa, leave_one_out=True)
    print("### Mode 3: leave-one-out, CALIBRATED (isotonic) + abstention")
    print(loo_cal.to_pretty())
    print(
        f"Calibration collapses the overconfidence: leave-one-out ECE "
        f"{loo.summary['ece']:.2f} -> {loo_cal.summary['ece']:.2f}. On this synthetic "
        "corpus the withheld answers aren't recoverable from the rest, so the "
        "calibrated system correctly abstains rather than answering confidently and "
        "wrong (abstention "
        f"{loo.summary['abstention_rate']:.0%} -> {loo_cal.summary['abstention_rate']:.0%}). "
        "On a real corpus with genuine cross-answer overlap the same calibrator learns "
        "a non-trivial map — answering the recoverable questions and abstaining on the "
        "rest (see the calibration unit tests for that behaviour on controlled data).\n"
    )

    print("Two example answers (note the citations, confidence, and staleness):")
    _, holdout = split_holdout(qa, settings.holdout_fraction)
    for qa_pair in holdout[:2]:
        _print_answer(qa_pair.question.splitlines()[0], bf.ask(qa_pair.question))

    _write_report(
        args.report,
        {
            "closed_book": closed.to_dict(),
            "leave_one_out": loo.to_dict(),
            "leave_one_out_calibrated": loo_cal.to_dict(),
        },
    )
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    print(f"Fetching closed issues from {args.repo} (max {args.max_issues})...")
    docs, qa = ingest_repo(args.repo, settings.github_token, max_issues=args.max_issues)
    out = Path(args.out)
    n_docs = write_jsonl(out / "docs.jsonl", docs)
    n_qa = write_jsonl(out / "eval.jsonl", qa)
    print(f"Wrote {n_docs} documents -> {out / 'docs.jsonl'}")
    print(f"Wrote {n_qa} QA pairs   -> {out / 'eval.jsonl'}")
    if args.db:
        with SqliteKnowledgeBase(args.db) as kb:
            kb.upsert_documents(docs)
            kb.upsert_qa_pairs(qa)
        print(f"Persisted to {args.db} (incremental upsert — re-run to update in place)")
    if n_qa == 0:
        print("No QA pairs found. Try a repo with question-style issues and maintainer replies.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    bf = _engine_from_args(args, settings)
    _print_answer(args.question, bf.ask(args.question))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.db:
        with SqliteKnowledgeBase(args.db) as kb:
            docs = kb.documents()
            qa = kb.qa_pairs()
        bf = BusFactor.from_documents(docs, settings=settings, use_dense=args.dense)
    else:
        qa = load_qa_pairs(args.eval)
        bf = BusFactor.from_jsonl(args.docs, settings=settings, use_dense=args.dense)
    if args.calibrate:
        bf.fit_calibration(qa, leave_one_out=args.leave_one_out)
    report = bf.evaluate(qa, leave_one_out=args.leave_one_out)
    print(report.to_pretty())
    _write_report(args.report, report.to_dict())
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    with SqliteKnowledgeBase(args.db) as kb:
        print(json.dumps(kb.stats(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bus-factor", description=__doc__)
    p.add_argument("--version", action="version", version=f"bus-factor {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    p.add_argument("--dense", action="store_true", help="enable dense retrieval if installed")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run end-to-end on bundled sample data (offline)")
    d.add_argument("--report", default="reports/demo_report.json")
    d.set_defaults(func=cmd_demo)

    i = sub.add_parser("ingest", help="scrape a public repo into docs + eval set")
    i.add_argument("repo", help="owner/name, e.g. simonw/datasette")
    i.add_argument("--out", default="data")
    i.add_argument("--max-issues", type=int, default=100, dest="max_issues")
    i.add_argument("--db", default="", help="also persist into this SQLite knowledge base")
    i.set_defaults(func=cmd_ingest)

    a = sub.add_parser("ask", help="answer a question from a knowledge base")
    a.add_argument("question")
    a.add_argument("--docs", default="data/docs.jsonl")
    a.add_argument("--db", default="", help="load from this SQLite knowledge base instead")
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval", help="score the answerer against a holdout set")
    e.add_argument("--docs", default="data/docs.jsonl")
    e.add_argument("--eval", default="data/eval.jsonl")
    e.add_argument("--db", default="", help="load docs + QA from this SQLite knowledge base")
    e.add_argument("--report", default="reports/eval_report.json")
    e.add_argument(
        "--leave-one-out",
        action="store_true",
        dest="leave_one_out",
        help="withhold each question's own answer (stricter generalization test)",
    )
    e.add_argument(
        "--calibrate",
        action="store_true",
        help="fit + apply isotonic confidence calibration with abstention",
    )
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("stats", help="summarize a persisted knowledge base")
    s.add_argument("--db", required=True, help="path to the SQLite knowledge base")
    s.set_defaults(func=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    configure_logging(getattr(args, "verbose", False))
    try:
        return int(args.func(args))
    except BusFactorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
