"""Fusion strategies compared in Paper 5."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from prresearch.p5_fusion.schema import Event


class Alert:
    __slots__ = ("key", "camera_id", "entity_type", "entity_id", "ts", "counter", "members")

    def __init__(self, key, event: Event) -> None:
        self.key = key
        self.camera_id = event.camera_id
        self.entity_type = event.entity_type
        self.entity_id = event.entity_id
        self.ts = event.ts
        self.counter = 1
        self.members = [event]

    def absorb(self, event: Event) -> None:
        self.counter += 1
        self.members.append(event)


class NaiveOr:
    """One alert per detection. The behaviour operators describe as alert fatigue."""

    name = "naive_or"

    def run(self, events: Sequence[Event]) -> list[Alert]:
        return [Alert((e.event_id,), e) for e in events]


class PerModalityDedup:
    """Collapse repeats within a modality only, the common VMS behaviour."""

    name = "per_modality_dedup"

    def __init__(self, window_s: float = 120.0) -> None:
        self.window_s = window_s

    def run(self, events: Sequence[Event]) -> list[Alert]:
        open_by_key: dict[tuple, Alert] = {}
        out: list[Alert] = []
        for e in sorted(events, key=lambda x: x.ts):
            key = (e.camera_id, e.modality, e.entity_id)
            cur = open_by_key.get(key)
            if cur is not None and e.ts - cur.ts <= self.window_s:
                cur.absorb(e)
                continue
            a = Alert(key, e)
            open_by_key[key] = a
            out.append(a)
        return out


class ConfidenceVoting:
    """Emit only when k modalities agree inside the window. Precision over recall."""

    name = "confidence_voting"

    def __init__(self, window_s: float = 120.0, votes: int = 2) -> None:
        self.window_s = window_s
        self.votes = votes

    def run(self, events: Sequence[Event]) -> list[Alert]:
        buckets: dict[tuple, list[Event]] = defaultdict(list)
        out: list[Alert] = []
        for e in sorted(events, key=lambda x: x.ts):
            key = (e.camera_id, e.entity_id)
            buf = buckets[key]
            buf[:] = [x for x in buf if e.ts - x.ts <= self.window_s]
            buf.append(e)
            if len({x.modality for x in buf}) >= self.votes:
                a = Alert(key, buf[0])
                for x in buf[1:]:
                    a.absorb(x)
                out.append(a)
                buf.clear()
        return out


class EntityAgnosticCollapse:
    """The proposed method: one predicate for every entity type and modality.

    Two observations collapse when they share a camera and resolve to the same
    entity, within the collapse window, regardless of which modality produced
    them. Cross-modal links (a plate seen by ANPR and the same vehicle seen by
    the object detector) resolve to the same entity id by the registry's
    cross-modal key, so no union type and no per-modality rule is needed.
    """

    name = "entity_agnostic_collapse"

    def __init__(self, window_s: float = 120.0) -> None:
        self.window_s = window_s

    def run(self, events: Sequence[Event]) -> list[Alert]:
        open_by_key: dict[tuple, Alert] = {}
        out: list[Alert] = []
        for e in sorted(events, key=lambda x: x.ts):
            key = (e.camera_id, e.entity_type, e.entity_id)
            cur = open_by_key.get(key)
            if cur is not None and e.ts - cur.ts <= self.window_s:
                cur.absorb(e)
                continue
            a = Alert(key, e)
            open_by_key[key] = a
            out.append(a)
        return out


def score(alerts: Sequence[Alert], total_incidents: int) -> dict:
    """Two failure modes, measured separately.

    Under-collapsing costs the operator attention: several alerts for one
    incident. Over-collapsing costs evidence: two distinct incidents merged into
    one alert, so the second one is never surfaced. A method is only good if it
    drives the first down without pushing the second up.
    """
    seen: dict[int, int] = defaultdict(int)
    masked = 0
    for a in alerts:
        incidents = {m.truth_incident for m in a.members}
        masked += len(incidents) - 1
        for inc in incidents:
            seen[inc] += 1
    covered = len(seen)
    redundant = sum(v - 1 for v in seen.values())
    return {
        "alerts_emitted": len(alerts),
        "incidents_covered": covered,
        "incident_recall": covered / total_incidents if total_incidents else float("nan"),
        "incidents_masked_by_over_collapse": masked,
        "distinct_incident_recall": (covered - masked) / total_incidents if total_incidents else float("nan"),
        "redundant_alerts": redundant,
        "alerts_per_incident": len(alerts) / (covered or 1),
    }
