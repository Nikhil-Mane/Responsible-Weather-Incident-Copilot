"""Turning an assessment into a filtered retrieval query."""

from __future__ import annotations

from typing import TYPE_CHECKING

from copilot.domain.models import Hazard
from copilot.retrieval.hybrid import RetrievalQuery

if TYPE_CHECKING:
    from copilot.domain.rules import Assessment

_TOPIC_BY_HAZARD: dict[Hazard, str] = {
    Hazard.INTENSE_RAINFALL: "flood",
    Hazard.RISING_RIVER_LEVEL: "flood",
    Hazard.ACTIVE_FLOOD_WARNING: "flood",
    Hazard.RAPID_RIVER_RISE: "flood",
    Hazard.HEAVY_SNOWFALL: "winter",
    Hazard.LOW_VISIBILITY: "winter",
    Hazard.HIGH_WIND: "wind",  # no approved wind playbook exists, so nothing is retrieved for it
    Hazard.EXTREME_HEAT: "heat",
    Hazard.HOT_ROAD_SURFACE: "heat",
}


# Vocabulary of each topic, added to the query so a hazard such as "hot road surface" still
# reaches the heat playbook even though its wording differs from the playbook's.
_TOPIC_TERMS = {
    "flood": "flood rainfall river drainage",
    "winter": "snow ice winter visibility road",
    "heat": "heat temperature cooling hydration",
}


COMMUNICATIONS_TOPIC = "communications"


DATA_QUALITY_TOPIC = "data_quality"


def build_query(
    assessment: Assessment,
    event: str,
    *,
    quality_failed: bool = False,
    allowed_ids: frozenset[str] | None = None,
    top_k: int = 3,
) -> RetrievalQuery:
    """Turn an assessment into a retrieval query with metadata filters.

    A hazard-free case yields an empty topic set (nothing to retrieve). Poor-quality input
    retrieves only the data-quality standard so the review reason can be cited.
    """
    if quality_failed:
        return RetrievalQuery(
            text="source confidence timestamp freshness data quality hold review",
            topics=frozenset({DATA_QUALITY_TOPIC}),
            allowed_ids=allowed_ids,
            top_k=top_k,
        )
    topics = {_TOPIC_BY_HAZARD[h] for h in assessment.hazards}
    if topics and (assessment.flood_risk or assessment.requires_incident_command_review):
        topics.add(COMMUNICATIONS_TOPIC)
    terms = [event, *(h.value for h in assessment.hazards)]
    terms.extend(_TOPIC_TERMS[t] for t in sorted(topics) if t in _TOPIC_TERMS)
    if COMMUNICATIONS_TOPIC in topics:
        terms.append("draft alert approval incident commander readiness")
    return RetrievalQuery(
        text=" ".join(terms),
        topics=frozenset(topics),
        allowed_ids=allowed_ids,
        top_k=top_k,
    )
