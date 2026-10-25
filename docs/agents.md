# Agents approach

How the agentic parts of the copilot are scoped, what they may and may not do, and why. Background:
[ADR 0001](adr/0001-deterministic-core-with-bounded-agents.md) (bounded agents),
[ADR 0002](adr/0002-langgraph-spine.md) (LangGraph spine) and
[ADR 0003](adr/0003-notification-boundary.md) (notification boundary).

## Principle: agents write, code decides

| Decision | Made by | Agent involvement |
|---|---|---|
| Is the input usable (source, freshness, confidence, geo, horizon, status)? | Deterministic quality gate | None |
| Which hazards exist, and how severe? | `domain/rules.py` thresholds | None |
| Which approved playbooks apply? | Retrieval (BM25 + vector, fusion, rerank) | None |
| Wording of readiness actions and the draft alert | **Agents** (or the rule-based drafter) | Drafts only |
| Are the drafted actions cited and grounded? Is the alert a `DRAFT:`? | Guardrails | None |
| Route: `human_review`, `operational_briefing` or `data_review` | LangGraph conditional edges | None |
| Approve, hold, send | Incident commander + signed token + notifier | None, and agents cannot reach it |

Risk, routing and approval are never agent decisions, so the safety properties hold with any model
or with none (`COPILOT_DRAFTER=rules`, the offline default).

## Three drafter implementations, one contract

All implement the `Drafter` protocol in [drafting/base.py](../src/copilot/drafting/base.py)
(`draft_actions`, `draft_alert`), so the graph does not know which one is running.

| Mode (`COPILOT_DRAFTER`) | Framework | Where | Use |
|---|---|---|---|
| `rules` | none | `drafting/rule_based.py` | Deterministic, offline, the evaluation baseline and the fallback |
| `lcel` | LangChain LCEL | `agents/chains.py`, `agents/llm_drafter.py` | `prompt \| model.with_structured_output(schema)`, one chain per step |
| `crew` | CrewAI | `agents/crew/` | Role-based drafting with scoped agents |
| `auto` | | `agents/factory.py` | LCEL when Azure is configured, otherwise rules |

## CrewAI roles

Two sequential crews, built per call in [agents/crew/crews.py](../src/copilot/agents/crew/crews.py).

```
Planning crew:   Hazard Analyst --(summary)--> Response Planner --> ActionPlan (schema-validated)
Comms crew:      Comms Drafter --> AlertDraft (schema-validated)
```

| Role | Scope | Input | Output |
|---|---|---|---|
| **Hazard Analyst** | Explain which hazards are present and why the severity is what it is | Verified facts only (rendered snapshot and assessment) | Two-sentence summary, passed to the planner as context |
| **Response Planner** | Turn approved playbooks into short readiness actions | Retrieved playbooks plus the analyst's summary | `ActionPlan`: actions, each with playbook citations |
| **Comms Drafter** | Draft the internal local-team alert | Verified facts plus the already-vetted actions | `AlertDraft`: text that must start with `DRAFT:` |

Constraints applied to every agent ([agents/crew/runtime.py](../src/copilot/agents/crew/runtime.py)):

* `tools=[]` and `allow_delegation=False`: roles can reason and write, nothing else. There is no send,
  notify or dispatch capability to misuse.
* `max_iter=2`, `memory=False`, `cache=False`, `tracing=False`, and CrewAI telemetry disabled before import,
  so incident data leaves the process only for the configured model provider.
* Output is a Pydantic schema (`output_pydantic`); a crew that does not return one raises a retryable
  `ProviderError`.
* Feed free text is sanitised and delimited as untrusted data before it reaches a prompt.

## Why this decomposition

* Analysis, planning and communication are different jobs with different inputs. Giving the planner only
  playbooks and the drafter only vetted actions limits what each can get wrong and keeps prompts small.
* The Comms Drafter runs **after** action guardrails, so the alert can only mention actions that already
  passed citation and grounding checks.
* A single agent would be simpler, which is why the LCEL mode exists. CrewAI is kept because the
  case study asks for scoped role orchestration, and the cost is measured, not hidden: more model calls
  per case (see the evaluation latency and token figures).

## Validation of agent output (outside the agents)

Performed in [workflow/graph.py](../src/copilot/workflow/graph.py) and
[safety/guardrails.py](../src/copilot/safety/guardrails.py) regardless of which drafter ran:

1. **Schema:** invalid structure is retried with jittered backoff behind a circuit breaker.
2. **Citation:** every action must cite a playbook that was actually retrieved; uncited actions are dropped.
3. **Grounding:** every action must textually overlap its cited playbook (`MIN_GROUNDING_OVERLAP`).
4. **Alert contract:** must be marked `DRAFT:`, name the event and affected area, and not claim to be sent.
5. **Fallback:** any `ProviderError`, or a rejected alert, switches to the rule-based drafter and records
   `fallback` and the reason in the trace. If nothing grounded survives for a flood alert, the case is
   routed to human review instead.

## Orchestration and the human boundary

LangGraph owns the workflow: `validate_input` -> (`hold_for_data_review` | `assess_hazards` ->
`retrieve_playbooks` -> `draft_actions` -> `draft_alert` -> `route_case` -> `human_review` |
`operational_briefing`). `human_review` calls `interrupt()` and the checkpointed run resumes only with a
commander's decision. Sending is **not** a graph node: the notifier is a separate component that the agent
packages cannot import (enforced by `tests/test_architecture.py`) and accepts only an expiring token bound
to the exact alert text.

## Observability of agent work

Each step is recorded in the briefing `trace` (including `fallback` status and reason) and the audit log.
With Langfuse configured, crew-mode agents and their model calls arrive in the same trace as the
workflow steps ([observability/crewai_bridge.py](../src/copilot/observability/crewai_bridge.py)).

## Evaluating agents

`python eval/run_eval.py` runs the same labeled cases through whichever drafter is configured, so a
model or crew can be compared with the deterministic baseline on route, alert, approval, citation and
groundedness accuracy, plus latency, token use and estimated cost. Use `--min-accuracy` to relax the
floors (route, alert, approval, citation, grounded, failure path) when evaluating a model. The default
floors are 100%, which is what the deterministic drafter must meet. Safety does not depend on these
floors: routing, approval and sending are code paths the drafter cannot influence.

## Known limits

* Agents add latency and cost for wording only; the deterministic drafter produces an equivalent structure.
* Strict grounding rejects some valid paraphrases; the fix is a fallback, not a failure.
* Crew mode needs the optional `agents` extra and is not exercised by the default offline run.
* Roles do not use tools or memory by design; adding either would need a new ADR and guardrail review.
