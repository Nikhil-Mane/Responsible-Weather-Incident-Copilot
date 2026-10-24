"""Shared fixtures: a trusted flood snapshot and a fixed clock."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from copilot.core.config import Settings, get_settings
from copilot.data.extract import SOURCE_FILES, extract_all
from copilot.domain.models import WeatherSnapshot
from copilot.retrieval.hybrid import HybridRetriever
from copilot.retrieval.playbooks import load_playbooks

NOW = datetime(2026, 9, 1, 9, 5, tzinfo=UTC)

# Settings the project reads. Tests must not inherit them from the developer's shell or `.env`.
_PROJECT_ENV_PREFIXES = ("COPILOT_", "AZURE_OPENAI_", "LANGFUSE_")


@pytest.fixture(autouse=True)
def _isolate_from_developer_settings(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep tests offline and deterministic, whatever is in the shell or in `.env`.

    Without this, a developer with real Azure or Langfuse settings would make the suite slow,
    cost tokens and fail. Live checks (`@pytest.mark.azure`) opt out and use the real settings.
    """
    if request.node.get_closest_marker("azure"):
        return
    for name in list(os.environ):
        if name.startswith(_PROJECT_ENV_PREFIXES):
            monkeypatch.delenv(name)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()


TRUSTED_FLOOD: dict[str, Any] = {
    "city": "Northbridge",
    "event": "River flood warning",
    "severity": "severe",
    "snowfall_rate_cm_per_hour": 0,
    "wind_gust_kmh": 35,
    "temperature_c": 8,
    "road_surface_temperature_c": 7,
    "visibility_m": 700,
    "rainfall_mm_per_hour": 42,
    "river_level_m": 4.9,
    "flood_warning": True,
    "forecast_rainfall_mm": 35,
    "river_rise_cm_per_hour": 14,
    "affected_areas": ["Riverside", "Old Town"],
    "road_closures": ["Harbor Road"],
    "observation_timestamp": "2026-09-01T09:00:00+00:00",
    "forecast_horizon_hours": 6,
    "source_id": "official-river-gauge-northbridge-07",
    "source_confidence": 0.96,
    "latitude": 51.5074,
    "longitude": -0.1278,
    "geo_zone": "Riverside-01",
    "population_at_risk": 1840,
    "critical_assets": ["Riverside Care Home", "Northbridge Substation"],
    "drainage_capacity_percent": 82,
    "incident_status": "active",
}


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def generated_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Extract the supplied PDFs once per session into a temporary directory."""
    if not all((DATA_DIR / name).is_file() for name in SOURCE_FILES.values()):
        pytest.skip("supplied PDFs are not present in data/")
    out = tmp_path_factory.mktemp("generated")
    extract_all(DATA_DIR, out)
    return out


@pytest.fixture(scope="session")
def real_retriever(generated_dir: Path):  # type: ignore[no-untyped-def]
    return HybridRetriever(load_playbooks(generated_dir / "operational_playbooks.json"))


@pytest.fixture
def now() -> datetime:
    return NOW


@pytest.fixture
def make_snapshot():  # type: ignore[no-untyped-def]
    def _make(**overrides: Any) -> WeatherSnapshot:
        return WeatherSnapshot.model_validate({**TRUSTED_FLOOD, **overrides})

    return _make


@pytest.fixture
def make_weather():  # type: ignore[no-untyped-def]
    """Weather-only snapshot with no flood or provenance fields, severity low by default."""

    def _make(**overrides: Any) -> WeatherSnapshot:
        base: dict[str, Any] = {"city": "Northbridge", "event": "Test", "severity": "low"}
        return WeatherSnapshot.model_validate({**base, **overrides})

    return _make
