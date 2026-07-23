"""Run the larger synthetic benchmark and print the three-mode comparison.

    python scripts/synthetic_benchmark.py

Deterministic: the corpus and the numbers reproduce exactly on any machine.
"""

from __future__ import annotations

from bus_factor.benchmark import load_benchmark, run_benchmark


def main() -> int:
    docs, qa = load_benchmark()
    print(f"Synthetic benchmark: {len(docs)} knowledge documents, {len(qa)} ground-truth QA pairs")
    print("(clustered issues share knowledge; singletons are unique)\n")

    reports = run_benchmark()
    labels = {
        "closed_book": "Closed-book (answer present)",
        "leave_one_out": "Leave-one-out, RAW (answer withheld)",
        "leave_one_out_calibrated": "Leave-one-out, CALIBRATED + abstention",
    }
    header = f"{'mode':<40}{'acc':>7}{'cover':>8}{'sel-acc':>9}{'ECE':>7}"
    print(header)
    print("-" * len(header))
    for key, label in labels.items():
        s = reports[key].summary
        print(
            f"{label:<40}{s['accuracy']:>6.0%}{s['coverage']:>8.0%}"
            f"{s['selective_accuracy']:>9.0%}{s['ece']:>7.2f}"
        )
    raw = reports["leave_one_out"].summary
    cal = reports["leave_one_out_calibrated"].summary
    print(
        f"\nUnder leave-one-out, calibration + abstention lifts accuracy-when-it-answers "
        f"from {raw['accuracy']:.0%} to {cal['selective_accuracy']:.0%} and cuts ECE from "
        f"{raw['ece']:.2f} to {cal['ece']:.2f}, by abstaining on the {1 - cal['coverage']:.0%} "
        f"of questions it can't recover from the rest of the corpus (coverage "
        f"{cal['coverage']:.0%}). That is selective prediction working: it trades some "
        f"coverage for answers you can trust."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
