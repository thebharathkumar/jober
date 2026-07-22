# Bus Factor

**Organizational memory that survives turnover.**

When an experienced person leaves a team, the company loses years of judgment
that was never written down: why a system was built the way it was, which
tradeoffs were considered and rejected, how to actually get things done. Bus
Factor captures that knowledge while the person is still around, then answers
questions the way the evidence shows *they* would — **with citations, a
confidence score, and the age of the evidence on every answer** — and, most
importantly, **measures how well it works.**

> The name is the point. [*Bus factor*](https://en.wikipedia.org/wiki/Bus_factor)
> is the number of people who can get hit by a bus before a project stalls. This
> raises it.

It is not a clone of a person. You cannot clone a person, and any project that
claims to is lying. This is a *knowledge-continuity system*: it makes captured
knowledge retrievable, faithfully reproduced, and — the part almost every RAG
demo skips — **verified against ground truth.**

---

## The one number that matters

Anyone can build retrieval over Slack and call it done. The thing that turns a
toy into an engineering artifact is proving it works. Bus Factor ships an
evaluation harness that grades the system against real question/answer pairs and
reports more than a single vanity metric:

Running `bus-factor demo` on the bundled sample corpus (offline, no API key):

| Mode | Accuracy | token-F1 | Retrieval hit-rate | ECE (calibration) |
|------|---------:|---------:|-------------------:|------------------:|
| **Closed-book** (answer is in the corpus) | 100% | 0.95 | 100% | **0.03** |
| **Leave-one-out** (answer withheld) | 0% | 0.20 | 0% | **0.77** |

Read those two rows together — that contrast *is* the result:

- **Closed-book** measures whether captured knowledge is retrievable and
  faithfully reproduced. It is well-calibrated (ECE 0.03): when the system is
  confident, it is right.
- **Leave-one-out** withholds each question's own answer and forces the system
  to respond from the person's *other* recorded knowledge. This is the honest
  generalization test, and it exposes a real weakness the harness is designed to
  catch: **naive retrieval confidence is overconfident under distribution
  shift** (ECE jumps to 0.77) — BM25 still finds lexically-similar documents when
  the true answer is absent, and reports high confidence anyway.

Surfacing that gap instead of hiding it is the entire reason evals exist.
Calibrating confidence so the system abstains when it truly doesn't know is the
[top roadmap item](docs/ROADMAP.md).

*(Numbers above are from the tiny bundled fixture and are illustrative of the
harness, not a benchmark claim. Point `bus-factor ingest` at a real repo to
generate a real eval set — see below.)*

---

## Quickstart

No dependencies, no API key, no network — the core runs on the Python standard
library alone.

```bash
git clone https://github.com/thebharathkumar/jober && cd jober
make demo          # end-to-end: ingest -> retrieve -> answer -> evaluate
```

Ask a question against a captured knowledge base, and see the provenance:

```bash
python -m bus_factor.cli ingest simonw/datasette --out data   # builds data/docs.jsonl
python -m bus_factor.cli ask "How do I enable full-text search?" --docs data/docs.jsonl
```

Every answer comes back like this — never bare text:

```
A: Set the DATAKIT_CACHE environment variable to any path and datakit will use it
   for all cached artifacts. It takes effect at import time.
   confidence=0.94  freshest-evidence=383d old
   sources:
     - Re: How do I change the cache directory?  (383d old)  https://github.com/acme/datakit/issues/104#issuecomment-1
```

### Generate a real eval set from a public repo

The insight the ingest layer operationalizes: **a maintainer's closed issues are
a pre-built evaluation set.** Someone asks a question, a maintainer answers, the
issue closes — that is a real question paired with a real authoritative answer,
thousands of them, for free. Authoritative answers are identified by GitHub's
`author_association` (`OWNER` / `MEMBER` / `COLLABORATOR`), not by string-matching
usernames.

```bash
export GITHUB_TOKEN=...           # optional, raises the rate limit
python -m bus_factor.cli ingest simonw/datasette --out data --max-issues 300
python -m bus_factor.cli eval --docs data/docs.jsonl --eval data/eval.jsonl
python -m bus_factor.cli eval --docs data/docs.jsonl --eval data/eval.jsonl --leave-one-out
```

### Turn on the real model

With an API key set, the answerer uses a Claude model and the judge grades
semantically instead of lexically; without it, both fall back to deterministic
offline implementations so evals always run.

```bash
export ANTHROPIC_API_KEY=...
python -m bus_factor.cli demo
```

---

## Architecture

Four layers, each testable and swappable in isolation. Full detail in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
  ingest/   closed issues ──> authoritative answers = knowledge Documents
            (author_associaton)                     + ground-truth QA pairs
                │
  memory/   hybrid retrieval (BM25 + optional dense), fused with RRF
            every hit carries its Source (url, author, date)
                │
  agent/    grounded answerer: retrieve ──> answer WITH
            citations + confidence + staleness   (never bare text)
                │
  eval/     ── THE DIFFERENTIATOR ──
            grade vs held-out QA: accuracy, F1, retrieval-hit-rate,
            calibration (ECE), abstention; closed-book AND leave-one-out
```

Three design decisions worth calling out, because they are where the judgment is:

1. **Knowledge is what the person *authored*, not what was asked of them.** The
   ingest layer indexes maintainers' *answers* as knowledge and treats the
   askers' questions purely as eval inputs. Indexing the questions as knowledge
   both pollutes the corpus and makes an extractive answerer echo the prompt
   back — a subtle modeling bug this design avoids by construction.
2. **Separate retrieval from generation.** Retrieval ranks with the issue title
   weighted in; generation is fed only the authored body, so the answerer can't
   parrot the question that titles are derived from.
3. **The core has zero dependencies.** BM25 is hand-rolled (~40 lines); dense
   retrieval and the LLM are optional accelerators that *degrade*, never crash.
   That is what lets `make demo` produce the same number on any machine, which
   is the whole point of a number you want people to trust.

---

## Ethics

The clickbait framing ("clone your engineer before they quit") is deliberately
provocative, but the system is built to be defensible, not creepy. See
[ETHICS.md](ETHICS.md). In short:

- **Consent is first-class.** In a real deployment, ingestion is opt-in and the
  subject reviews what is captured.
- **Provenance over impersonation.** Every answer cites public, checkable
  sources. The system reproduces *documented* knowledge; it does not pretend to
  be a person, and it clones no voice or likeness.
- **The bundled sample data is fully synthetic** (a fictional `datakit` project),
  so nothing in this repo impersonates a real individual.

---

## Project layout

```
src/bus_factor/
  ingest/    github_issues.py   # closed issues -> knowledge + eval set (pure transforms + I/O)
  memory/    bm25.py store.py    # dependency-free BM25 + hybrid RRF store
             embeddings.py       # optional dense retriever (degrades if absent)
  agent/     answerer.py         # retrieve -> grounded answer with provenance
             provider.py         # Anthropic + offline extractive fallback
  eval/      harness.py          # closed-book + leave-one-out, full report
             metrics.py judge.py dataset.py
  cli.py     config.py           # argparse CLI, env-driven settings
tests/                           # 20 tests, no network required
```

## Status & roadmap

This is a working foundation, built in the open as a portfolio project. It is
honest about what is done and what is next — see [docs/ROADMAP.md](docs/ROADMAP.md).
The headline next step is **confidence calibration** (close the ECE gap the
leave-one-out eval exposes), followed by a persona/voice layer via a small
MLX LoRA fine-tune kept strictly separate from the facts.

## License

MIT — see [LICENSE](LICENSE).
