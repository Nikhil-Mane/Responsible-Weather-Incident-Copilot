# ADR 0002: LangGraph as the workflow spine

**Status:** accepted

## Context

The workflow branches (bad data, review, approval), must pause for a human, and must resume the same
run later, possibly in a different process. A linear chain cannot express that cleanly.

## Decision

Use LangGraph with a typed, JSON-compatible state and a SQLite checkpointer. Quality and risk are
conditional edges; `human_review` calls `interrupt()` and resumes with the commander's decision.
LangChain LCEL composes the drafting chain; CrewAI defines the scoped drafting roles. The notification
step is **not** a graph node.

## Consequences

* A paused run survives a restart (tested by building a second workflow on the same database).
* State is plain JSON, so checkpoints are inspectable and do not depend on pickled objects.
* The resumed node re-executes up to the `interrupt()` call, so nodes before it must be idempotent; the
  review node is written that way.
* Three frameworks are used because each does a different job; none is needed for the safety rules.
