# Security

## Reporting

This is a portfolio project. If you find a security issue, please open a GitHub
issue describing it (omit any real secrets), or contact the maintainer.

## Handling of secrets

- No secrets are stored in the repo. `ANTHROPIC_API_KEY` and `GITHUB_TOKEN` are
  read from the environment only; `.env` is git-ignored and `.env.example` holds
  placeholders.
- Tokens are never logged. Structured logging records request outcomes and
  counts, not headers or credentials.

## Handling of untrusted input

- **Repo slugs are validated** against `^owner/name$` before being placed in an
  API URL, to prevent path injection into the GitHub request.
- **Ingested content is untrusted.** Issue and comment bodies come from arbitrary
  users. They are treated as data: indexed and cited, never executed, and never
  interpolated into a shell or a query. Answers surface this content with a link
  back to its source so a human can judge it.
- **Malformed data fails safe.** Invalid JSONL rows are validated and skipped
  with a logged warning rather than crashing a batch.

## Network

- Outbound calls go only to `api.github.com` (ingest) and, when a key is set,
  the Anthropic API. Both use HTTPS. The ingest client has bounded timeouts and
  retry limits.

## Deployment note

If you deploy this against non-public knowledge, see [ETHICS.md](ETHICS.md):
consent-gated ingestion, subject review/redaction, and cited answers are part of
the intended operating model.
