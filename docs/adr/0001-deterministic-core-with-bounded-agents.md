# ADR 0001: Deterministic core with bounded agents

**Status:** accepted

## Context

The case study wants agentic GenAI, but also hard safety boundaries: no autonomous dispatch, no use of
invalid data, an auditable decision path. A free-form autonomous agent cannot guarantee those, is
non-deterministic, and is hard to test at threshold boundaries.

## Decision

Severity, hazard detection, data-quality checks, routing, approval and sending are **deterministic
code**. Models (via LCEL chains or CrewAI roles) only draft wording: readiness actions and the alert
text. Agents get no tools, so they cannot act. Their output is schema-validated and then checked for
citations, textual grounding and the `DRAFT:` contract before it is used. If a model fails, the
rule-based drafter produces the same structure.

## Consequences

* Safety properties hold with any model, or none. The whole system runs and is tested offline.
* Threshold behaviour is exact and testable (property-based tests at every boundary).
* The model cannot improve a risk decision, only the prose around it. That is deliberate.
* A little duplicated capability (rule-based and model drafters) in exchange for a guaranteed fallback.
