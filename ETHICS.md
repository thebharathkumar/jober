# Ethics & responsible use

Bus Factor uses a deliberately provocative framing — "capture an employee's
knowledge before they leave." Provocative framings invite misuse, so the design
takes a clear position on what this is and is not.

## What it is

A **knowledge-continuity system**: it makes *documented, authored* knowledge
retrievable and reproducible, and attaches provenance to every answer so a human
can verify it.

## What it is not

- **Not a clone of a person.** It reproduces recorded knowledge; it does not
  model a personality, and it clones no voice, face, or likeness.
- **Not a replacement for consent.** Capturing someone's work product to answer
  as them is something you do *with* a person, not *to* them.
- **Not a surveillance tool.** It ingests knowledge artifacts (answers, docs),
  not behavioural tracking.

## Principles the system is built around

1. **Consent is first-class.** In any real deployment, ingestion is opt-in and
   the subject reviews and can redact what is captured before it is used.
2. **Provenance over impersonation.** Every answer cites checkable sources. The
   goal is "here is what the record says, and where," not "trust me, I'm them."
3. **Public data, handled as public.** When run against a public repository, the
   system indexes information the author already published, and always links
   back to it. Even so, a courtesy heads-up to a named individual before
   building a public demo on their corpus is the right thing to do.
4. **Staleness is surfaced, not hidden.** Knowledge decays; an answer that
   leans on old evidence says so.

## The sample data in this repo

The bundled corpus (`src/bus_factor/_data/sample_issues.json`) is **entirely
fictional** — a made-up `datakit` library and invented usernames. Nothing in
this repository impersonates or reproduces a real person's knowledge.

## If you deploy this

- Get explicit, revocable consent from the subject.
- Give the subject a review-and-redact pass over captured knowledge.
- Keep answers cited and let users see sources.
- Do not present the system's output as if a specific human personally said it.
- Treat the confidence score as a hint, not a guarantee — and note it is
  currently uncalibrated under distribution shift (see the roadmap).
