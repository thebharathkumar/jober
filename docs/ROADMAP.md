# Roadmap

Built in phases, depth-first, where each phase ships something demonstrable and
de-risks the next. Status is honest: this repo is Phases 0–1 plus the eval spine
of Phase 2.

## Phase 0 — Build the eval set first ✅

Turn a source of authored answers into ground-truth QA pairs *before* building
anything that answers. Define success before optimizing for it.

- [x] GitHub closed-issue ingestion with `author_association` answer detection
- [x] Pure, unit-tested transforms separated from network I/O
- [x] Deterministic holdout split

## Phase 1 — Memory layer (retrieval with provenance) ✅

- [x] Dependency-free BM25
- [x] Hybrid store with RRF fusion; optional dense retriever
- [x] Every retrieval result carries its `Source` (url, author, date)
- [x] Grounded answerer: citations + confidence + staleness on every answer

## Phase 2 — Evaluation ✅ (spine) / 🚧 (calibration)

- [x] Closed-book and leave-one-out harness
- [x] accuracy, token-F1, ROUGE-L, retrieval-hit-rate, abstention
- [x] Calibration: ECE + per-confidence-band accuracy table
- [x] Pluggable judge (lexical offline / LLM semantic)
- [ ] **Confidence calibration — the top priority.** The leave-one-out eval
      shows retrieval confidence is overconfident under distribution shift
      (ECE ≈ 0.77 vs 0.03 closed-book). Plan: fit isotonic regression on a
      held-out calibration split, add an *answerability* signal (retrieval-score
      floor + margin), and make the answerer abstain below threshold. Target:
      leave-one-out ECE < 0.15 and a non-trivial correct-abstention rate.
- [ ] Confidence-interval / bootstrap on the headline metrics
- [ ] Regression gate in CI (fail the build if accuracy drops > X%)

## Phase 3 — Skill layer (agentic)

Move from *recall* to *action*. Give the answerer tools (search the repo, read
code, run tests) so it can triage a new, open issue end to end rather than only
answering from memory.

- [ ] Tool-use loop with a bounded action budget
- [ ] "Resolve an open issue" evaluation track (did the proposed fix apply?)

## Phase 4 — Persona / voice layer (optional fine-tune)

A single MLX LoRA on a 7–8B model for *tone only*, kept strictly separate from
the facts. The deliverable is an **ablation** showing that
fine-tune-for-voice + RAG-for-facts beats fine-tune-for-everything — the study
matters more than the weights.

- [ ] MLX LoRA training script (runs on an M-series Mac)
- [ ] Voice-similarity metric distinct from factual accuracy
- [ ] Ablation chart

## Phase 5 — Interfaces

- [ ] Minimal web UI showing the **Bus Factor score** and answer provenance
- [ ] Structured exit-interview agent that extracts decision records (the tacit
      knowledge that never makes it into docs)
- [ ] Consent + audit trail for corporate deployment

## Known limitations (today)

- Retrieval confidence is uncalibrated under distribution shift (Phase 2).
- The bundled sample corpus is tiny and synthetic; real numbers require running
  `ingest` against a real repo.
- No chunking yet — long documents are indexed whole. Fine for issue-sized text,
  needs revisiting for long-form docs.
