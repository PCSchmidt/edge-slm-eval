"""The only functions a model-emitted call may ever reach. Pure, local, deterministic."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

# Simulated local sensor table: no network, no filesystem.
SENSORS: dict[str, str] = {
    "thermal-01": "OK",
    "thermal-02": "DEGRADED",
    "radar-07": "OFFLINE",
    "acoustic-03": "OK",
    "gps-01": "OK",
}


def get_sensor_status(sensor_id: str) -> str:
    return SENSORS.get(sensor_id, "UNKNOWN")


def azimuth_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    # Round first, then wrap: a tiny negative bearing must not round up to exactly 360.0.
    return round(math.degrees(math.atan2(y, x)), 3) % 360.0


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * 6371.0088 * math.asin(math.sqrt(a)), 3)


def battery_hours(capacity_wh: float, load_w: float) -> float:
    if load_w <= 0:
        raise ValueError("load_w must be positive")
    return round(capacity_wh / load_w, 3)


@dataclass(frozen=True)
class ToolSpec:
    fn: Callable[..., object]
    arg_types: tuple[type, ...]  # positional signature; ints are accepted where float is expected


REGISTRY: dict[str, ToolSpec] = {
    "get_sensor_status": ToolSpec(get_sensor_status, (str,)),
    "azimuth_deg": ToolSpec(azimuth_deg, (float, float, float, float)),
    "distance_km": ToolSpec(distance_km, (float, float, float, float)),
    "battery_hours": ToolSpec(battery_hours, (float, float)),
}
