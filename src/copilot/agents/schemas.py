"""Schemas a model must return."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


class PlannedAction(BaseModel):
    """One action as proposed by the model."""

    text: str = Field(min_length=3, max_length=300)
    citations: list[str] = Field(min_length=1, max_length=5)


class ActionPlan(BaseModel):
    """Schema the model must return for the action step."""

    actions: list[PlannedAction] = Field(max_length=12)


class AlertDraft(BaseModel):
    """Schema the model must return for the alert step."""

    text: str = Field(min_length=10, max_length=1200)
