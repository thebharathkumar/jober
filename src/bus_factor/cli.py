"""Command-line entry point.

    bus-factor demo                      end-to-end on bundled sample data (offline)
    bus-factor ingest <owner/repo>       scrape a public repo -> docs + eval set
    bus-factor ask   "<question>"        answer from a docs.jsonl, with citations
    bus-factor eval                      score the answerer against a holdout set

Built on argparse so the CLI itself adds zero dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

from bus_factor.agent.answerer import Answerer
from bus_factor.config import Settings
from bus_factor.eval.dataset import load_documents, load_qa_pairs, split_holdout
from bus_factor.eval.harness import make_leave_one_out_factory, run_eval
from bus_factor.eval.judge import get_judge
from bus_factor.ingest.github_issues import ingest_repo, transform_repo
from bus_factor.memory.store import MemoryStore
from bus_factor.models import Document, write_jsonl


def _load_sample_pipeline() -> tuple[list[Document], list]:
    raw = json.loads(
        resources.files("bus_factor").joinpath("_data/sample_issues.json").read_text("utf-8")
    )
    pairs = [(entry["issue"], entry.get("comments", [])) for entry in raw]
    return transform_repo(pairs)


def _print_answer(question: str, ans) -> None:
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


def cmd_demo(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    docs, qa = _load_sample_pipeline()

    store = MemoryStore(rrf_k=settings.rrf_k, use_dense=args.dense)
    store.add(docs)
    answerer = Answerer(store, settings)
    judge = get_judge(settings)

    print(
        f"Loaded {len(docs)} knowledge documents and {len(qa)} ground-truth QA pairs "
        f"from bundled sample data."
    )
    _, holdout = split_holdout(qa, settings.holdout_fraction)

    # Mode 1 — closed-book: the answer IS in the store. Measures retrieval +
    # faithful reproduction of captured knowledge (can it recall what it learned).
    closed = run_eval(holdout, judge, answerer=answerer)
    print("\n### Mode 1: closed-book (captured knowledge is retrievable)")
    print(closed.to_pretty())

    # Mode 2 — leave-one-out: each question's own answer is withheld, so the
    # system must answer from the person's OTHER knowledge. Measures generalization
    # and, crucially, whether the trust layer backs off when it truly doesn't know.
    loo_factory = make_leave_one_out_factory(docs, settings, use_dense=args.dense)
    loo = run_eval(holdout, judge, answerer_factory=loo_factory)
    print("### Mode 2: leave-one-out (answer withheld — the honest generalization test)")
    print(loo.to_pretty())
    print(
        "Read the two together. Closed-book is well-calibrated (low ECE): captured "
        "knowledge is faithfully recalled. Leave-one-out withholds the answer and "
        "exposes overconfidence under distribution shift (ECE rises sharply) — naive "
        "retrieval confidence stays high even when the right source is absent. "
        "Surfacing that gap is exactly what the harness is for; calibrating confidence "
        "so the system abstains when it truly doesn't know is the top roadmap item.\n"
    )

    print("Two example answers (note the citations, confidence, and staleness):")
    for qa_pair in holdout[:2]:
        _print_answer(qa_pair.question.splitlines()[0], answerer.answer(qa_pair.question))

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"closed_book": closed.to_dict(), "leave_one_out": loo.to_dict()}, indent=2),
        encoding="utf-8",
    )
    print(f"\nFull report written to {out}")
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
    if n_qa == 0:
        print("No QA pairs found. Try a repo with question-style issues and maintainer replies.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    docs = load_documents(args.docs)
    store = MemoryStore(rrf_k=settings.rrf_k, use_dense=args.dense)
    store.add(docs)
    answerer = Answerer(store, settings)
    _print_answer(args.question, answerer.answer(args.question))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    docs = load_documents(args.docs)
    qa = load_qa_pairs(args.eval)
    store = MemoryStore(rrf_k=settings.rrf_k, use_dense=args.dense)
    store.add(docs)
    answerer = Answerer(store, settings)
    judge = get_judge(settings)
    _, holdout = split_holdout(qa, settings.holdout_fraction)
    if args.leave_one_out:
        factory = make_leave_one_out_factory(docs, settings, use_dense=args.dense)
        report = run_eval(holdout, judge, answerer_factory=factory)
    else:
        report = run_eval(holdout, judge, answerer=answerer)
    print(report.to_pretty())
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"Full report written to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bus-factor", description=__doc__)
    p.add_argument("--dense", action="store_true", help="enable dense retrieval if installed")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run end-to-end on bundled sample data (offline)")
    d.add_argument("--report", default="reports/demo_report.json")
    d.set_defaults(func=cmd_demo)

    i = sub.add_parser("ingest", help="scrape a public repo into docs + eval set")
    i.add_argument("repo", help="owner/name, e.g. simonw/datasette")
    i.add_argument("--out", default="data")
    i.add_argument("--max-issues", type=int, default=100, dest="max_issues")
    i.set_defaults(func=cmd_ingest)

    a = sub.add_parser("ask", help="answer a question from a docs.jsonl")
    a.add_argument("question")
    a.add_argument("--docs", default="data/docs.jsonl")
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval", help="score the answerer against a holdout set")
    e.add_argument("--docs", default="data/docs.jsonl")
    e.add_argument("--eval", default="data/eval.jsonl")
    e.add_argument("--report", default="reports/eval_report.json")
    e.add_argument(
        "--leave-one-out",
        action="store_true",
        dest="leave_one_out",
        help="withhold each question's own answer (stricter generalization test)",
    )
    e.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
