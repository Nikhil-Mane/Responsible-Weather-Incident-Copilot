# ADR 0005: Optional services degrade; they never block a safe result

**Status:** accepted

## Context

Azure OpenAI, Azure embeddings and Langfuse are external services that can be missing, slow or down.
An operations tool must still produce a correct, safe briefing.

## Decision

* **Chat model:** timeouts, bounded retries with jittered backoff and a circuit breaker; any failure
  (including content-filter blocks and invalid structured output) falls back to the rule-based drafter,
  and the reason is written to the trace.
* **Embeddings:** if the corpus or a query cannot be embedded, retrieval continues with BM25 only and
  reports `vector_available = 0` on each result.
* **Langfuse:** all observer calls are wrapped; failures are logged and swallowed.
* **A hazard with no matching playbook:** produce a warning and no actions rather than inventing
  guidance; send the case to a person if an alert would otherwise be drafted without guidance.

## Consequences

* The system has no hard runtime dependency on any external service, and all of this is tested with fakes; the Azure paths were also exercised against the live service.
* Operators can tell degraded from normal operation (trace status `fallback` / `no_evidence`, warnings,
  Langfuse levels), so degradation is visible rather than silent.
* Live-service behaviour is verified by the opt-in `pytest -m azure` checks and by manual runs against the
  live services (see the README limitations). Silent degradation is the risk: a wrong Azure API version once
  made every call 404 and the system quietly used the rule-based drafter, which is why `fallback` is surfaced
  in the trace and the runbook explains how to read it.
