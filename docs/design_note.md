# Design note

A short note on data quality, security, privacy, model and provider choice, and production
deployment. See the [README](../README.md) for architecture and the [ADRs](adr/) for the decisions.

## Data quality

* **Typed boundary.** Input is parsed into a frozen Pydantic snapshot with `extra="forbid"`, range
  limits, no NaN or infinity, and timezone-aware timestamps. Size (256 KiB) is capped before parsing and
  nesting (12 levels) is checked, iteratively, immediately after it.
* **Quality gate** (PLAYBOOK-DATA-005), pure functions with an injected clock and policy: traceable
  source from an approved prefix, observation freshness (default 60 min), no future timestamps, source
  confidence (default 0.7), geo zone or coordinates, forecast horizon (default 48 h) and an active
  incident status. Any finding routes the case to `data_review`: no hazards, actions or alert are
  produced, and the findings are listed so an analyst can fix the feed.
* **Missing is not zero.** Unreported fields stay `None`, so an absent rainfall reading cannot look like
  "no rain".
* **Provenance of our own data.** The supplied PDFs are the only data source and are never modified;
  extraction is reproducible and covered by integrity tests (24 scenarios, 5 playbooks, spot-checked
  threshold cells). Anything derived (the evaluation's provenance wrapper and failure cases) is labeled.

## Security

* **No autonomous send.** The agent workflow has no send tool and cannot import the notifier. The
  notifier contains no network code; a test fails if one appears.
* **Approvals are signed, scoped and single-purpose.** HMAC-SHA256 tokens bind run ID and the SHA-256 of
  the exact alert text, expire (default 15 min), and are accepted only from the incident-commander role.
  Editing the alert after approval invalidates the token. Sending is idempotent per run and alert and is
  serialised with a file lock (a Windows-specific lock race was found by stress testing and fixed).
* **Tamper-evident audit.** Append-only JSONL; each entry commits to the previous entry's hash and
  `copilot audit-verify` checks the chain. Secrets are redacted before they are written.
* **RBAC in code, not in the UI.** Roles gate running, approving and playbook visibility (the response
  team cannot read the data-quality standard); the checks live in the functions that do the work.
* **LLM-specific.** Free text from feeds is sanitised and delimited as untrusted data; models get no
  tools; all model output is schema-validated, then checked for citations, textual grounding and the
  `DRAFT:` contract. A compromised or confused model can at worst produce a rejected draft, because
  severity, routing and sending are not model decisions.
* **Hygiene.** Secrets are `SecretStr` and environment-only; no `eval`, no pickle; file paths are
  resolved and size-checked; run IDs are validated so they cannot escape the state directory; CrewAI
  telemetry is disabled; `bandit` and `pip-audit` run in CI.

### Dependency audit (`pip-audit`)

The audit is part of `make check` and CI. Its first run on the original pins found advisories in the
LangGraph SQLite checkpointer (SQL injection) and checkpoint deserialization, which this system uses for
pause and resume, so the dependency ranges were raised (LangGraph 1.x, LangChain 1.x, CrewAI 1.x) and the
full suite re-run. Seven advisories are **deliberately ignored**, all transitive and all in features this
project does not use:

| Advisory | Package | Why it does not apply |
|---|---|---|
| PYSEC-2026-311, -3813, -3814, -3815 | chromadb (via CrewAI) | Concern the Chroma server and its authorization providers. No fix is published. We run no Chroma server, and crews are built with `memory=False` |
| CVE-2026-112940, -112913, -112907 | filelock (via LiteLLM) | Concern `ReadWriteLock` and the Windows soft locks. We use our own lock (`core/locking.py`); a fix needs filelock 4.x, which LiteLLM currently forbids |

Re-check these on each dependency upgrade and remove the exceptions as soon as fixes are installable.
## Privacy

The supplied data is synthetic and contains no personal data. The design still minimises: briefings use
area-level facts (zones, population at risk as a count), logs are redacted, and only the facts needed for
drafting are sent to the model. For real data: keep personal data out of prompts, set retention on the
audit log and run store, and avoid free-text fields that could carry names.

## Model and provider choice

* **Azure OpenAI** was chosen because the organisation already holds an account, which brings data
  residency options, a no-training-on-customer-data commitment, private networking and enterprise
  identity. A model from the gpt-4o class or newer is recommended for reliable structured output; the
  deployment name is configuration, not code.
* **The model is replaceable and optional.** The LLM only drafts wording; the deterministic drafter
  produces a complete, valid briefing with no provider at all. Provider timeouts, throttling, content
  filter blocks and invalid output are retried with jittered backoff, behind a circuit breaker, then fall
  back to rules, with the reason recorded in the trace.
* **Embeddings** (hybrid retrieval) use a separate Azure deployment and degrade to lexical search.
* **Cost and latency** are measured by the evaluation (token usage and an estimated cost when prices are
  configured).

## Production deployment

What exists now is a proof of concept with production-grade engineering; what a deployment adds:

| Concern | Here | In production |
|---|---|---|
| Identity | `--role` / `--actor` asserted by the caller | SSO (OIDC), roles from the identity provider, approver identity taken from the session, MFA for commanders |
| Secrets | `.env` | Managed secret store (Key Vault); token signing key rotated |
| Notification | Simulated adapter | Queue-backed notification service behind the same `send` contract; signed approvals verified by the service; delivery receipts |
| State | Local SQLite and files | Managed database for checkpoints and runs; append-only (WORM) storage for the audit log |
| Data feeds | JSON files extracted from PDFs | Authenticated feed ingestion with source allow-lists and schema contracts |
| Operations | Structured logs, optional Langfuse (steps, agents, model calls, send, eval scores) | Central logging, metrics and alerting (rejection rate, fallback rate, approval turnaround), SLOs, load testing |
| Delivery | Dockerfile, CI | Image scanning, signed images, staged rollout, rules-version change control with the evaluation as a release gate |
| Resilience | Retries, breaker, fallback | Multi-region provider fallback, graceful degradation drills |
