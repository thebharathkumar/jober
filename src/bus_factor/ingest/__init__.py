"""Ingestion: raw sources -> Documents (for memory) + QAPairs (for eval)."""

from bus_factor.ingest.github_issues import (
    build_qa_pair,
    ingest_repo,
    issue_to_document,
)

__all__ = ["build_qa_pair", "ingest_repo", "issue_to_document"]
