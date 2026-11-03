"""
Single source of truth for Odisha district names.

Parses the same GeoJSON that Leaflet uses for the dashboard heat map,
so the dropdown choices stay in lock-step with the map. Result is
cached at module import time; cost is paid once per process.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from django.conf import settings


GEOJSON_PATH = Path(settings.BASE_DIR) / "static" / "vendor" / "odisha.geojson"


@lru_cache(maxsize=1)
def get_odisha_districts() -> List[str]:
    """Return a sorted, deduped list of district names from the Odisha GeoJSON."""
    try:
        with open(GEOJSON_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    names = set()
    for feature in data.get("features", []):
        props = feature.get("properties", {}) or {}
        # Try the same property names the Leaflet code tries
        name = (
            props.get("district")
            or props.get("Dist_Name")
            or props.get("NAME_2")
            or props.get("DISTRICT")
            or props.get("dist_name")
            or ""
        )
        name = (name or "").strip()
        if name:
            names.add(name)
    return sorted(names)


def get_odisha_district_choices(include_blank: bool = True) -> List[Tuple[str, str]]:
    """Returns choices suitable for Django ChoiceField."""
    choices = [(d, d) for d in get_odisha_districts()]
    if include_blank:
        choices = [("", "— Select district —")] + choices
    return choices
