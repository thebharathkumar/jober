"""Bus Factor — organizational memory that survives turnover.

The package is layered so each concern can be tested and swapped independently:

    ingest/   raw sources (GitHub issues, docs) -> Documents + eval QA pairs
    memory/   retrieval over Documents, with provenance (hybrid BM25 + dense)
    agent/    grounded answerer: retrieve -> answer WITH citations/confidence
    eval/     the differentiator — measure agreement against ground truth

Nothing here imports a third-party package at module load time; optional
accelerators are imported lazily so the core always runs on the stdlib.
"""

__version__ = "0.1.0"
