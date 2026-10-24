# ADR 0004: The supplied PDFs are the only data source

**Status:** accepted

## Context

The scenario data, flood example and playbooks arrived as printed spreadsheets (PDF). Plain text
extraction scrambles column order and blank cells, playbook keyword cells are clipped by the next column,
and `Scenarios.pdf` has no text layer at all (vector outlines).

## Decision

Keep the PDFs unmodified and extract them with a reproducible script (`scripts/extract_data.py`) that
writes derived JSON to `data/generated/`, with a manifest of source hashes. Column assignment uses word
coordinates for the scenario table, column order for the single-row files, and character positions for
the playbooks. Integrity tests assert 24 labeled scenarios, 5 playbooks, and spot-checked threshold
cells. `Scenarios.pdf` is not used: it is only a threshold reference and the labeled dataset covers it.

## Consequences

* No invented replacement data; every derived value traces back to a supplied file.
* Clipped playbook keywords are recovered only as far as they were printed (flagged `keywords_clipped`).
  Retrieval also indexes titles and content, so recall is unaffected in practice (retrieval R@3 = 1.0).
* The supplied heatwave and winter files carry no provenance, so as supplied they are held for data
  review. The evaluation derives trusted copies and labels them as derived.
* Thresholds not given by the case study are inferred from the labels and documented in the README.
