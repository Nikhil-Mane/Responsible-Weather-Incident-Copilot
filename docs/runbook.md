# Runbook

Operational procedures. State lives under `COPILOT_STATE_DIR` (default `var/`).

| File | Content |
|---|---|
| `var/checkpoints.sqlite` | LangGraph checkpoints (paused and finished runs) |
| `var/runs/<run_id>.json` | Latest briefing per run |
| `var/audit.jsonl` | Hash-chained audit log |
| `var/notifications/sent.json` | Simulated sends (idempotency record) |
| `var/eval/report.json` | Latest evaluation report |

## Start a case and get it approved

1. `copilot run <incident.json> [--as-of <ISO time>]` prints the briefing. Exit code 4 means the case is
   waiting for the incident commander; 3 means it was held for data review.
2. The commander runs `copilot approve <run_id> --actor "<name>"` (or `copilot hold ...`). Approval prints
   an approval token that expires (default 15 minutes) and is bound to the exact alert text.
3. `copilot send <run_id> --token <token>` performs the **simulated** send. Re-running it is safe and
   returns the original receipt with `duplicate: true`.

## A run is stuck awaiting approval

`copilot status <run_id>`: if `awaiting_decision` is true, the commander has not decided. Decisions
survive restarts because the pause is checkpointed. To abandon the case, `copilot hold <run_id>`.

## A token expired or was lost

Run `copilot approve <run_id> --actor "<name>"` again. If the run is already approved this issues a fresh
token without changing the decision; if the alert was already sent, a new token still cannot cause a
second send.

## "the alert changed after it was approved"

The stored alert text no longer matches the one the token was issued for. Do not edit briefings by hand.
Re-run the incident and approve the new alert.

## Verify the audit log

`copilot audit-verify` (exit code 7 on failure). A failure names the first broken entry (edited content,
deleted entry or broken link). Treat it as a security incident: preserve `var/audit.jsonl` and
`var/checkpoints.sqlite`, and investigate before resuming operations.

## Data is held for review (exit code 3)

The briefing's `review_reasons` list every finding, for example `stale_observation`, `low_confidence`,
`missing_source`, `untrusted_source`, `inactive_incident`. Fix the feed (or confirm the source), then
re-run. Do not lower the quality policy to make a case pass; change `COPILOT_MAX_OBSERVATION_AGE_MINUTES`
and similar settings only through change control.

## Provider problems

* **HTTP 404 "Resource not found" in `fallback_reason`:** almost always `AZURE_OPENAI_API_VERSION` holding a
  model version (for example `2025-04-14`) instead of an API version (for example `2024-10-21`). Check the
  deployment name too (`AZURE_OPENAI_DEPLOYMENT` is the deployment's name, not the model's).
* **Langfuse:** needs SDK 5.x (`pip install -e ".[observability]"`), the three `LANGFUSE_*` settings, and
  `LANGFUSE_HOST` set to your region's URL. Traces appear a few seconds after a run. Do not set
  `OTEL_SDK_DISABLED`: the SDK is built on OpenTelemetry and it would switch tracing off.
  Traces contain prompts and model outputs (credential-looking strings are redacted).
* Trace steps with status `fallback` mean the model failed and the rule-based drafter was used; briefings
  are still valid. A sustained rise in fallbacks points at the provider (throttling, content filter,
  credentials). The circuit breaker opens after repeated failures and retries after about a minute.
* Embedding failures show as `vector_available = 0` and mean lexical-only retrieval.
* Langfuse failures only log a warning.

## Rotating the approval secret

Set a new `COPILOT_APPROVAL_SECRET`. Outstanding tokens stop verifying immediately; commanders re-issue
them with `copilot approve`. Nothing else depends on the secret.

## Changing a threshold (rules version)

1. Edit `RuleConfig` in `src/copilot/domain/rules.py` and bump `RULES_VERSION`.
2. Update or add the boundary tests in `tests/test_rules.py`.
3. Run `python eval/run_eval.py`; every gate must still pass. The version is recorded in every briefing.
4. Document the change in the README table.

## Before a release

`make check` (lint, types, tests with coverage, security scans), `python eval/run_eval.py`, and, if you
have credentials, `pytest -m azure`.
