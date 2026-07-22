"""CLI integration smoke tests, including the central error handling."""

from __future__ import annotations

import pytest

from bus_factor.cli import main
from bus_factor.memory.persistence import SqliteKnowledgeBase
from bus_factor.models import Document, Source


def _seed_db(path) -> None:
    with SqliteKnowledgeBase(path) as kb:
        kb.upsert_documents(
            [
                Document(
                    id="issue-1-answer",
                    text="Set DATAKIT_CACHE to change the cache directory at import time.",
                    source=Source(
                        id="issue-1-answer",
                        kind="issue",
                        title="Re: cache directory",
                        created_at="2025-01-01T00:00:00Z",
                    ),
                )
            ]
        )


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_demo_runs_and_writes_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["demo"]) == 0
    assert (tmp_path / "reports" / "demo_report.json").exists()


def test_stats_reads_knowledge_base(tmp_path, capsys):
    db = tmp_path / "kb.db"
    _seed_db(db)
    assert main(["stats", "--db", str(db)]) == 0
    assert '"documents"' in capsys.readouterr().out


def test_ask_from_db(tmp_path, capsys):
    db = tmp_path / "kb.db"
    _seed_db(db)
    assert main(["ask", "How do I change the cache directory?", "--db", str(db)]) == 0
    assert "DATAKIT_CACHE" in capsys.readouterr().out


def test_invalid_repo_slug_exits_one(capsys):
    # validate_repo raises IngestError, which main() catches -> exit 1, no traceback.
    assert main(["ingest", "not-a-valid-slug"]) == 1
    assert "error:" in capsys.readouterr().err
