"""A larger, *realistic* synthetic benchmark for the calibration story.

The bundled demo fixture is tiny and every withheld answer is unrecoverable, so
leave-one-out collapses to a clean 0% and calibration just learns "abstain on
everything". Real corpora aren't like that: a person answers the same topic many
times, so when one answer is withheld a *sibling* answer often still covers it.

This module generates that structure deterministically:

* **Clusters** — several issues about the same feature, whose answers share a
  distinctive identifier and a core fact. Withhold one and its siblings still
  answer it (leave-one-out RECOVERABLE, high retrieval confidence).
* **Singletons** — one-off questions with a unique identifier that appears in no
  other answer. Withhold it and nothing covers it (leave-one-out UNRECOVERABLE,
  low retrieval confidence).

That correlation — confident-and-right on clusters, unconfident-and-wrong on
singletons — is exactly what a confidence calibrator needs to become useful:
it learns to keep the clustered answers and abstain on the singletons.

Everything is deterministic (no wall-clock, no unseeded randomness) so the
reported numbers reproduce exactly.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from bus_factor.app import BusFactor
from bus_factor.config import Settings
from bus_factor.eval.harness import EvalReport
from bus_factor.ingest.github_issues import transform_repo
from bus_factor.models import Document, QAPair

_MAINTAINER = "nimbus-maintainer"
_BASE = date(2024, 1, 1)

_Q_TEMPLATES = [
    "How do I configure {w} in the Nimbus client?",
    "Where do I set the {w} options for Nimbus?",
    "Why is {w} not taking effect in Nimbus?",
    "Can I change {w} at runtime in Nimbus?",
    "What setting controls {w} in Nimbus?",
]

# Each cluster: a distinctive identifier, a topic, a shared core answer sentence
# (carries the identifier -> siblings match strongly), and per-issue (question,
# extra-sentence) pairs. The shared core is what makes a withheld answer
# recoverable from a sibling.
_CLUSTERS: list[dict[str, Any]] = [
    {
        "ident": "NIMBUS_TOKEN",
        "topic": "authentication",
        "core": (
            "Set the NIMBUS_TOKEN environment variable before calling connect(), and "
            "the Nimbus client authenticates every request with that token automatically."
        ),
        "items": [
            ("How do I authenticate the Nimbus client?", "It takes effect on the next connect() call."),
            ("Where do I put my Nimbus API token?", "In CI, store NIMBUS_TOKEN as a secret rather than committing it."),
            ("Why does Nimbus return 401 Unauthorized?", "A 401 almost always means NIMBUS_TOKEN is missing or expired."),
            ("How do I rotate my Nimbus credentials?", "Issue a new token in the dashboard and update NIMBUS_TOKEN."),
        ],
    },
    {
        "ident": "retry_policy",
        "topic": "retries and timeouts",
        "core": (
            "Configure retries by passing a retry_policy to the Nimbus client; it retries "
            "idempotent calls with exponential backoff up to the configured maximum."
        ),
        "items": [
            ("How do I make the Nimbus client retry failed requests?", "Pass retry_policy=RetryPolicy(max_attempts=5)."),
            ("Does Nimbus retry on network errors?", "Yes, the retry_policy covers transient network failures."),
            ("How do I set a request timeout in Nimbus?", "The retry_policy also carries a per-attempt timeout field."),
            ("Why isn't Nimbus retrying my request?", "A missing retry_policy means requests fail on the first error."),
        ],
    },
    {
        "ident": "NIMBUS_CACHE",
        "topic": "caching",
        "core": (
            "Point the NIMBUS_CACHE environment variable at a writable directory and Nimbus "
            "caches downloaded artifacts there, reusing them across runs."
        ),
        "items": [
            ("How do I change the Nimbus cache directory?", "NIMBUS_CACHE takes effect at import time."),
            ("Where does Nimbus store cached data?", "By default under the home cache dir, overridden by NIMBUS_CACHE."),
            ("How do I clear the Nimbus cache?", "Delete the NIMBUS_CACHE directory; it is rebuilt on demand."),
            ("Can I share a Nimbus cache between machines?", "Point NIMBUS_CACHE at a shared volume."),
        ],
    },
    {
        "ident": "log_level",
        "topic": "logging",
        "core": (
            "Control Nimbus logging with the log_level setting; set it to DEBUG to see every "
            "request and response the client makes."
        ),
        "items": [
            ("How do I enable debug logging in Nimbus?", "Set log_level='DEBUG' on the client."),
            ("Why is Nimbus so quiet by default?", "The default log_level is WARNING to stay quiet in notebooks."),
            ("How do I send Nimbus logs to a file?", "Combine log_level with a standard logging handler."),
            ("Can I silence Nimbus logging entirely?", "Set log_level='CRITICAL'."),
        ],
    },
    {
        "ident": "region",
        "topic": "regions",
        "core": (
            "Select where requests go with the region setting on the Nimbus client; each "
            "region is an isolated endpoint and data does not cross between them."
        ),
        "items": [
            ("How do I choose a Nimbus region?", "Pass region='us-east' when constructing the client."),
            ("Why can't I see my data in Nimbus?", "Data is region-scoped; check the region setting matches."),
            ("Does Nimbus replicate across regions?", "No, each region is isolated by design."),
            ("How do I move Nimbus data between regions?", "Export from the source region and import into the target."),
        ],
    },
    {
        "ident": "page_size",
        "topic": "pagination",
        "core": (
            "Nimbus list endpoints paginate with a cursor; set page_size to control the batch "
            "and follow next_cursor until it is null to read everything."
        ),
        "items": [
            ("How do I paginate Nimbus list results?", "Loop while next_cursor is not null."),
            ("What is the max page_size in Nimbus?", "page_size caps at 1000 per request."),
            ("Why do I only get 100 Nimbus results?", "100 is the default page_size; raise it explicitly."),
            ("How do I stream all Nimbus records?", "Follow the cursor with a small page_size to bound memory."),
        ],
    },
]

# Singletons: a distinctive identifier that appears in no other answer, and
# genuinely distinct wording, so a withheld singleton is unrecoverable from the
# rest of the corpus. Each answer is a full two clauses (>= 80 chars) so it
# survives ingest and doesn't collapse to a sibling.
_SINGLETONS: list[tuple[str, str]] = [
    ("How do I export Prometheus metrics from Nimbus?", "Enable the metrics_exporter and Nimbus serves a Prometheus scrape target on port 9464 with per-endpoint latency histograms."),
    ("Does Nimbus support gRPC transport?", "Set transport='grpc' and Nimbus multiplexes calls over a single HTTP/2 connection, falling back to REST when gRPC is unreachable."),
    ("How do I pin a Nimbus TLS certificate?", "Pass tls_pin set to the server certificate's sha256 fingerprint and Nimbus rejects any connection presenting a different certificate."),
    ("Can Nimbus run fully offline?", "Enable offline_mode and the client serves every read from the local cache and refuses network calls, raising OfflineError on a miss."),
    ("How do I set a custom user agent in Nimbus?", "Assign user_agent on the client and that string is attached to the User-Agent header of every outgoing request for attribution."),
    ("Does Nimbus support webhooks?", "Register a webhook_url and Nimbus delivers HMAC-signed event payloads to it whenever a resource you subscribed to changes state."),
    ("How do I compress Nimbus uploads?", "Set compression to zstd and the client negotiates it during the handshake, transparently falling back to gzip for older servers."),
    ("How do I batch writes in Nimbus?", "Open the write_batch context manager, enqueue records inside it, and Nimbus flushes the whole batch in a single round trip on exit."),
    ("Can I use Nimbus behind an HTTP proxy?", "The client reads the standard proxy environment variables and tunnels through CONNECT, so no explicit proxy option is required."),
    ("How do I validate a Nimbus schema locally?", "Run the schema check subcommand against your definition file and it reports backward-incompatible changes without any network access."),
    ("Does Nimbus deduplicate identical requests?", "Attach an idempotency_key to a mutating call and Nimbus collapses retries of the same key into one server-side effect for a day."),
    ("How do I throttle Nimbus client concurrency?", "Set max_inflight and the client queues additional calls once that many requests are already outstanding, smoothing burst load."),
    ("How do I subscribe to Nimbus change streams?", "Open a change_stream and iterate the events; on reconnect it resumes precisely from the last resume_token you persisted."),
    ("Can Nimbus encrypt the local cache at rest?", "Supply a local_key and every file the client writes to its cache directory is sealed with AES-256-GCM before touching disk."),
    ("How do I use a custom DNS resolver with Nimbus?", "Provide a resolver callable and Nimbus routes all hostname lookups through it, which is handy for split-horizon or test DNS."),
    ("How do I tune the Nimbus connection pool?", "Set pool_size to cap reused sockets and pool_ttl to close idle ones, trading memory for fewer TLS handshakes under load."),
    ("What request signing algorithm does Nimbus use?", "Every request is signed with Ed25519 by default, and you can supply a signer object to swap in an HSM-backed signing key."),
    ("How do I do a multipart upload in Nimbus?", "Call upload_multipart and the client splits the blob into parts, uploads them in parallel, and asks the server to assemble them."),
    ("How do I resume an interrupted Nimbus download?", "The first partial read returns a range_token; pass it back and the client resumes the byte stream exactly where it stopped."),
    ("How do I mask sensitive fields in Nimbus?", "Register field_mask patterns and any matching value is redacted inside the client before the payload is ever sent upstream."),
    ("How do I enable audit logging in Nimbus?", "Set an audit_sink callable and Nimbus emits a tamper-evident, hash-chained record of every mutating operation it performs."),
    ("Does Nimbus support feature flags?", "Read entitlements through the flags accessor; they are fetched once at startup and quietly refreshed when flag_ttl expires."),
    ("How do I run Nimbus against a sandbox?", "Set environment to sandbox and the client targets a disposable backend seeded with fixtures, so tests never touch production."),
    ("How does Nimbus handle clock skew?", "It measures the server time offset on connect and stamps requests accordingly, tolerating drift up to the max_clock_skew bound."),
    ("How do I add a dead-letter queue in Nimbus?", "Configure a dead_letter_url and any async event that exhausts its retries is parked there with its failure metadata for replay."),
    ("How do I apply backpressure in Nimbus?", "The producer side blocks once buffered bytes cross the high_watermark you set, so a slow consumer cannot exhaust client memory."),
    ("How do I rotate encryption keys in Nimbus?", "Add the new key to the keyring and Nimbus re-encrypts lazily on write while still decrypting old data with the retired key."),
    ("How do I trace a slow Nimbus request?", "Enable the tracing hook and each call emits an OpenTelemetry span with DNS, connect, and server-processing timings attached."),
    ("Does Nimbus support optimistic concurrency?", "Send the etag you last read as if_match and the server rejects the write with a conflict if the resource changed meanwhile."),
    ("How do I stream large results from Nimbus?", "Use the iter_rows helper, which pulls bounded windows and yields rows so a huge result set never has to fit in memory at once."),
]


def _issue(number: int, title: str, body: str, answer: str, day: int) -> dict[str, Any]:
    created = (_BASE + timedelta(days=day)).isoformat() + "T00:00:00Z"
    ans_created = (_BASE + timedelta(days=day + 1)).isoformat() + "T00:00:00Z"
    return {
        "issue": {
            "number": number,
            "title": title,
            "body": body,
            "html_url": f"https://github.com/acme/nimbus/issues/{number}",
            "user": {"login": f"user-{number}"},
            "created_at": created,
            "labels": [{"name": "question"}],
        },
        "comments": [
            {
                "body": answer,
                "author_association": "OWNER",
                "user": {"login": _MAINTAINER},
                "html_url": f"https://github.com/acme/nimbus/issues/{number}#issuecomment-1",
                "created_at": ans_created,
            }
        ],
    }


def generate_corpus(
    n_clusters: int = 12, cluster_size: int = 4, n_singletons: int = 0
) -> list[dict[str, Any]]:
    """Deterministically build the GitHub-issue-shaped benchmark corpus.

    The hand-written ``_CLUSTERS`` / ``_SINGLETONS`` seed it with realistic-reading
    content; the counts above pad it out to a stable size with schematic issues
    that carry the same recoverable-vs-unrecoverable structure via distinctive
    per-item identifiers. Bigger corpus -> a holdout large enough that the
    calibration numbers are stable rather than noisy.
    """
    entries: list[dict[str, Any]] = []
    number = 1000
    day = 0

    # 1) Realistic hand-written clusters (recoverable under leave-one-out).
    for cluster in _CLUSTERS:
        for question, extra in cluster["items"]:
            answer = f"{cluster['core']} {extra}"
            entries.append(_issue(number, question, "", answer, day))
            number += 1
            day += 5

    # 2) Schematic clusters for scale. Each shares a distinctive "widgetK" topic
    #    and "FLAG_K" identifier across its issues, so a withheld answer is
    #    strongly recovered from a sibling (high retrieval confidence, correct).
    for k in range(n_clusters):
        w = f"widget{k}"
        flag = f"FLAG_{k}"
        core = (
            f"Set {flag} to configure {w} in the Nimbus client; {flag} controls how "
            f"{w} behaves on every request and is read once at startup."
        )
        for j in range(cluster_size):
            q = _Q_TEMPLATES[j % len(_Q_TEMPLATES)].format(w=w)
            a = f"{core} For the {w} case {j}, set {flag} to the {w} profile value."
            entries.append(_issue(number, q, "", a, day))
            number += 1
            day += 5

    # 3) Hand-written singletons (unrecoverable under leave-one-out).
    for question, answer in _SINGLETONS:
        entries.append(_issue(number, question, "", answer, day))
        number += 1
        day += 5

    # 4) Schematic singletons for scale. Each "gadgetJ"/"OPT_J" identifier appears
    #    in exactly one answer, so withholding it leaves nothing to recover from
    #    (low retrieval confidence, wrong) — the case abstention should catch.
    for j in range(n_singletons):
        g = f"gadget{j}"
        opt = f"OPT_{j}"
        q = f"How do I enable {g} in Nimbus?"
        a = (
            f"Turn on {g} by setting {opt} in the Nimbus client; {g} is handled "
            f"entirely on the server side and needs no other local configuration."
        )
        entries.append(_issue(number, q, "", a, day))
        number += 1
        day += 5

    return entries


def load_benchmark() -> tuple[list[Document], list[QAPair]]:
    """Run the corpus through the real ingest transforms -> docs + eval pairs."""
    entries = generate_corpus()
    return transform_repo([(e["issue"], e["comments"]) for e in entries])


def run_benchmark(settings: Settings | None = None) -> dict[str, EvalReport]:
    """Produce the three-mode comparison used to demonstrate calibration."""
    settings = settings or Settings()
    docs, qa = load_benchmark()
    bf = BusFactor.from_documents(docs, settings=settings)
    closed = bf.evaluate(qa)
    loo_raw = bf.evaluate(qa, leave_one_out=True)
    bf.fit_calibration(qa, leave_one_out=True)
    loo_calibrated = bf.evaluate(qa, leave_one_out=True)
    return {"closed_book": closed, "leave_one_out": loo_raw, "leave_one_out_calibrated": loo_calibrated}
