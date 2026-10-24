# ADR 0003: The notification boundary is outside the agent workflow

**Status:** accepted

## Context

"Never send before approval" must not depend on a prompt being obeyed.

## Decision

Sending lives in a separate adapter (`safety/notify.py`). The graph never imports it and no agent has a send
tool. The adapter needs an HMAC-signed approval token that is bound to the run and to the SHA-256 of the
exact alert text, expires, and is accepted only from the incident-commander role. It is idempotent per
run and alert, serialised with a file lock, audited in a hash-chained log, and contains no network code
(a test asserts this and a second test runs it with sockets disabled).

## Consequences

* A manipulated model can at worst produce a draft that is rejected or ignored.
* Editing an alert after approval invalidates the approval.
* Replacing the simulation with a real channel means implementing the same `send` contract behind the
  same checks, not changing the workflow.
* Because approval and sending are separate steps, a lost response can be retried safely: a repeated send
  returns the original receipt flagged as a duplicate.
