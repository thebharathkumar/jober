# Architecture

Bus Factor is four layers with narrow interfaces. Each can be tested and
replaced without touching the others, which is what keeps the eval numbers
attributable to a specific component.

```
ingest  ->  memory  ->  agent  ->  eval
```

## Data model (`models.py`)

- **`Source`** — immutable provenance: id, kind, title, url, author, created_at.
  Frozen so a citation can never drift from what it pointed at.
- **`Document`** — a retrievable unit of knowledge plus its `Source`. Knows its
  own `age_days(as_of)` because staleness is a first-class concern.
- **`QAPair`** — a ground-truth question + the authoritative `reference_answer`.
  The system is judged on reproducing these under holdout.
- **`Answer`** — never bare text. Carries `citations`, `confidence`,
  `staleness_days`, and a `meta` dict of trust flags.

## Ingest (`ingest/github_issues.py`)

Turns a public repo's closed issues into knowledge + a ground-truth eval set.

**Key insight:** a maintainer's answered, closed issues *are* a labelled dataset.
The question is the issue; the label is the authoritative reply.

**Key modeling decision:** knowledge is what the subject *authored* (their
answers), not the questions asked of them. So only authoritative answers become
`Document`s. The asker's issue body lives on solely as an eval question. This
avoids two failure modes at once — corpus pollution with non-subject text, and
an extractive answerer that echoes the question back.

**Authoritative answer detection** uses GitHub's `author_association`
(`OWNER`/`MEMBER`/`COLLABORATOR`) rather than username string-matching, and
requires a minimum answer length and asker≠answerer.

**I/O is split from transforms.** `fetch_*` do network calls; `issue_to_document`,
`build_qa_pair`, and `transform_repo` are pure and fully unit-tested against
fixtures. That boundary is why the extraction logic is verifiable in CI with no
network — and thus why the downstream numbers can be trusted.

## Memory (`memory/`)

Hybrid retrieval, provenance-carrying.

- **`bm25.py`** — a dependency-free BM25 Okapi. Hand-rolled so the core needs no
  install. Lexical retrieval is essential for exact identifiers (error strings,
  flags, function names) that dense models blur.
- **`embeddings.py`** — optional dense retriever (`sentence-transformers`).
  Reports `available = False` and disables itself if the package or model cache
  is missing. An optional accelerator must degrade, not crash.
- **`store.py`** — fuses lexical and dense rankings with **Reciprocal Rank
  Fusion**. RRF needs no score calibration between the two retrievers, which
  matters because BM25 scores and cosine similarities are not comparable. The
  document title is weighted into the retrieval text (strong signal for
  issue-style content) but *not* passed to generation.

## Agent (`agent/`)

- **`provider.py`** — the `LLMProvider` contract with two implementations:
  `AnthropicProvider` (real model) and `ExtractiveProvider` (offline,
  deterministic; selects the answer sentences from the evidence). The extractive
  provider is rank-aware — it decays sentence scores by the retrieval rank of
  their source document to suppress top-k pollution — and echo-resistant: it
  drops sentences that are almost entirely question words.
- **`answerer.py`** — the pipeline: retrieve → build context → generate →
  attach citations, confidence, staleness. `confidence` is an explicit
  *heuristic proxy* from retrieval quality (a saturating function of the top
  lexical score plus a corroboration bonus). We do not assert it is calibrated;
  the eval harness *measures* whether it is.

## Eval (`eval/`) — the differentiator

- **`metrics.py`** — token-F1, ROUGE-L, normalized exact-match, and Expected
  Calibration Error. Multiple complementary metrics, because one headline number
  is easy to game and easy to misread.
- **`judge.py`** — `LexicalJudge` (F1 threshold, offline default) and `LLMJudge`
  (semantic, when a key is present). Every reported number is stamped with which
  judge produced it.
- **`dataset.py`** — deterministic holdout split by a stable hash of each QA id,
  so the split is identical on every machine with no seed threading.
- **`harness.py`** — runs the answerer over the holdout and reports accuracy,
  mean F1/ROUGE-L, **retrieval-hit-rate** (did the right source even get
  retrieved — separates retrieval failures from generation failures),
  abstention rate, and calibration (ECE + a per-band accuracy table). Supports
  two modes: **closed-book** (fixed store) and **leave-one-out** (each
  question's own answer withheld — the real generalization test).

## Persistence (`memory/persistence.py`)

`SqliteKnowledgeBase` is the durable source of truth, distinct from the
in-memory retrieval index. You ingest expensively once (network, rate limits),
persist here, and rebuild the cheap BM25 index from SQLite on start. Ingest is
**incremental**: documents upsert by primary key, so re-running against a repo
updates only what changed. SQLite is deliberate — zero-ops, single-file, ACID,
already in the stdlib, and swapping in Postgres later is confined to this one
module.

## The facade (`app.py`)

`BusFactor` is the one public object. It composes the store, answerer, and eval
into `ask()` / `evaluate()` and offers `from_documents` / `from_jsonl` /
`from_sqlite` constructors so the source of truth is a caller choice. The CLI is
a thin shell over it, and it dogfoods the same API a library user would.

## Cross-cutting concerns

- **`config.py`** — a single `Settings` object, env-driven, **validated at
  construction** (`__post_init__` raises `ConfigError` on out-of-range values).
  One printable object describes the whole system's behaviour.
- **`errors.py`** — a typed hierarchy rooted at `BusFactorError`, so callers can
  catch our failures distinctly from bugs and the CLI can exit cleanly.
- **`logging_config.py`** — library-safe logging: modules stay quiet
  (`get_logger`); only the application (`configure_logging`, called by the CLI)
  attaches a handler. Imported into a larger service, Bus Factor does not hijack
  the root logger.

## Resilience in the ingest client

Network I/O is isolated in `github_issues.py` and hardened: exponential backoff
with retries, rate-limit detection (raises `RateLimitError` with the reset time
when `X-RateLimit-Remaining` hits 0, honors `Retry-After`), `404 → RepoNotFound`,
5xx → retry, and repo-slug validation to keep untrusted input out of the request
URL. The pure transforms remain separate and unit-tested, so extraction logic is
verified without touching the network.

## Confidence calibration (`calibration.py`)

Raw retrieval confidence is a heuristic and, as the leave-one-out eval shows,
overconfident under distribution shift. `IsotonicCalibrator` fixes the numbers:
it fits a monotonic map from raw score → empirical correctness using a
hand-rolled **weighted Pool Adjacent Violators** algorithm (no dependency).
Isotonic regression is the right choice because it assumes only monotonicity
(higher retrieval confidence should not mean lower accuracy) and otherwise lets
the data set the shape.

The fit happens on a **held-out calibration split** (`BusFactor.fit_calibration`),
in the *same regime* it will be applied to (leave-one-out data to correct
leave-one-out overconfidence). Once installed on the `Answerer`, confidence is
mapped through the calibrator and the answerer **abstains** below
`abstain_threshold` — but only when calibrated, because abstaining on an
untrustworthy raw score would just relocate the problem. The eval reports
`coverage` and `selective_accuracy` so the coverage/accuracy trade is explicit.
On the bundled corpus this takes leave-one-out ECE from 0.77 to ~0.00.

## Why not fine-tune the facts?

A recurring amateur move is to fine-tune a model on the corpus and call the
weights "the clone." That fails in three ways: it hallucinates confidently, it
cannot cite, and it cannot be updated when knowledge changes. Bus Factor keeps
**facts in retrieval** (updatable, citable, staleness-aware) and reserves
fine-tuning for a *separate, optional* persona/voice layer (see the roadmap).
Judgment about when to fine-tune — and when not to — is the point.
