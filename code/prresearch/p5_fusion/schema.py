"""Paper 5: the entity-agnostic event record and the collapse predicate.

Mirrors app/services/analyse.py::_base_event and app/services/matcher.py, where
a single record shape carries vehicle, person and region-occupancy detections,
and one 120 s predicate collapses repeats of the same entity on the same camera
into one alert with a counter.

The nine fields that matter for fusion are kept here; the deployed record adds
crop and audit fields that do not affect the collapse decision.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    event_id: str
    entity_type: str      # vehicle | person | region
    entity_id: str        # plate, gallery id, or region id
    camera_id: str
    ts: float
    modality: str         # anpr | objects | faces | occupancy
    confidence: float
    bbox: tuple[int, int, int, int]
    truth_incident: int   # evaluation only; never read by a fusion method


def optimal_window_s(fov_depth_m: float, speed_limit_kmh: float, margin: float = 1.0) -> float:
    """The window the patent derives from geometry: dwell time in the field of view.

    A vehicle at the permitted speed crosses the field of view in
    fov_depth / speed seconds. Two observations further apart than that cannot
    be the same pass, so collapsing beyond it merges distinct incidents.
    """
    speed_ms = max(speed_limit_kmh, 1.0) / 3.6
    return margin * fov_depth_m / speed_ms
