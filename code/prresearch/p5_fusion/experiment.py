"""Paper 5 experiments.

E5.1  Alert-fatigue reduction and incident recall against three baselines.
E5.2  Collapse-window sweep: the reduction/recall trade, and where the geometric
      window sits on that curve.
E5.3  The derived window against the empirically best window, per camera, as
      field of view depth and speed limit vary.
E5.4  Cross-modal contribution: what the entity-agnostic key buys over
      per-modality dedup, by share of dual-tagged incidents.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prresearch.p5_fusion.collapse import (
    ConfidenceVoting,
    EntityAgnosticCollapse,
    NaiveOr,
    PerModalityDedup,
    score,
)
from prresearch.p5_fusion.schema import Event, optimal_window_s
from prresearch.seeds import rng
from prresearch.traces import build_estate

RESULTS = Path(__file__).resolve().parents[2] / "results"
MODALITY_OF = {"vehicle": ("anpr", "objects"), "person": ("faces", "objects"), "region": ("occupancy",)}


def synth_detections(
    n_incidents: int,
    dual_share: float,
    seed_name: str,
    n_cameras: int = 200,
    revisit_share: float = 0.25,
    revisit_gap_s: tuple[float, float] = (90.0, 900.0),
):
    """Each incident is one entity passing one camera; it emits several frames.

    A share of incidents are revisits: the same entity returning to the same
    camera after a gap. Those are distinct incidents an operator must see, so a
    collapse window longer than the gap destroys recall. Without revisits the
    window sweep is degenerate and every window looks equally good.
    """
    estate = build_estate(n_cameras, "grid", seed_name="p5")
    cams = estate.cameras
    g = rng(seed_name)
    events: list[Event] = []
    meta = []
    eid = 0
    prior: list[tuple[object, str, str, float]] = []
    for inc in range(n_incidents):
        if prior and float(g.random()) < revisit_share:
            cam, etype, entity, t_prev = prior[int(g.integers(0, len(prior)))]
            t0 = t_prev + float(g.uniform(*revisit_gap_s))
        else:
            cam = cams[int(g.integers(0, len(cams)))]
            etype = str(g.choice(["vehicle", "person", "region"], p=[0.55, 0.35, 0.10]))
            entity = f"{etype[:3].upper()}{int(g.integers(0, 40000)):05d}"
            t0 = float(g.uniform(0, 86400))
        prior.append((cam, etype, entity, t0))
        dwell = optimal_window_s(cam.fov_depth_m, cam.speed_limit_kmh)
        n_frames = int(np.clip(g.poisson(4) + 1, 1, 14))
        mods = list(MODALITY_OF[etype])
        dual = len(mods) > 1 and float(g.random()) < dual_share
        for _ in range(n_frames):
            ts = t0 + float(g.uniform(0, dwell))
            mod = mods[0] if not dual else mods[int(g.integers(0, len(mods)))]
            events.append(
                Event(
                    event_id=f"E{eid:08d}",
                    entity_type=etype,
                    entity_id=entity,
                    camera_id=cam.camera_id,
                    ts=ts,
                    modality=mod,
                    confidence=float(np.clip(g.beta(6, 2), 0, 1)),
                    bbox=(0, 0, 40, 40),
                    truth_incident=inc,
                )
            )
            eid += 1
        meta.append({"incident": inc, "camera_id": cam.camera_id, "dwell_s": dwell, "dual": dual, "t0": t0})
    return events, meta, estate


def e5_1_baselines(n_incidents: int = 12000) -> dict:
    events, meta, _ = synth_detections(n_incidents, dual_share=0.45, seed_name="p5:main")
    rows = []
    for method in (NaiveOr(), PerModalityDedup(120.0), ConfidenceVoting(120.0, 2), EntityAgnosticCollapse(120.0)):
        s = score(method.run(events), n_incidents)
        s["method"] = method.name
        rows.append(s)
    base = next(r for r in rows if r["method"] == "naive_or")["alerts_emitted"]
    for r in rows:
        r["alert_reduction_vs_naive"] = 1.0 - r["alerts_emitted"] / base
    return {
        "experiment": "E5.1_fusion_baselines",
        "detections": len(events),
        "incidents": n_incidents,
        "dual_tagged_incidents": sum(1 for m in meta if m["dual"]),
        "rows": rows,
    }


def e5_2_window_sweep(n_incidents: int = 12000) -> dict:
    events, meta, _ = synth_detections(n_incidents, dual_share=0.45, seed_name="p5:main")
    median_dwell = float(np.median([m["dwell_s"] for m in meta]))
    rows = []
    for w in (0, 5, 15, 30, 60, 120, 180, 240, 480, 900, 1800):
        s = score(EntityAgnosticCollapse(float(w)).run(events), n_incidents)
        s["window_s"] = w
        rows.append(s)
    return {
        "experiment": "E5.2_collapse_window_sweep",
        "median_geometric_window_s": median_dwell,
        "deployed_window_s": 120,
        "rows": rows,
    }


def e5_3_derived_window() -> dict:
    """Per-camera geometric window vs. the single deployed constant."""
    _, meta, estate = synth_detections(6000, dual_share=0.45, seed_name="p5:geom")
    by_id = estate.by_id()
    rows = []
    for cam in estate.cameras[:12]:
        rows.append(
            {
                "camera_id": cam.camera_id,
                "fov_depth_m": round(cam.fov_depth_m, 1),
                "speed_limit_kmh": cam.speed_limit_kmh,
                "geometric_window_s": round(optimal_window_s(cam.fov_depth_m, cam.speed_limit_kmh), 2),
            }
        )
    allw = [optimal_window_s(c.fov_depth_m, c.speed_limit_kmh) for c in estate.cameras]
    del by_id
    return {
        "experiment": "E5.3_geometric_window_per_camera",
        "min_s": float(np.min(allw)),
        "median_s": float(np.median(allw)),
        "max_s": float(np.max(allw)),
        "deployed_constant_s": 120,
        "cameras_where_constant_over_collapses": int(sum(1 for w in allw if w < 120)),
        "cameras": len(allw),
        "rows": rows,
    }


def e5_4_cross_modal(n_incidents: int = 12000) -> dict:
    rows = []
    for dual in (0.0, 0.15, 0.30, 0.45, 0.70, 1.0):
        events, _, _ = synth_detections(n_incidents, dual_share=dual, seed_name=f"p5:dual:{dual}")
        per = score(PerModalityDedup(120.0).run(events), n_incidents)
        ent = score(EntityAgnosticCollapse(120.0).run(events), n_incidents)
        rows.append(
            {
                "dual_tagged_share": dual,
                "per_modality_alerts": per["alerts_emitted"],
                "entity_agnostic_alerts": ent["alerts_emitted"],
                "extra_reduction": 1.0 - ent["alerts_emitted"] / (per["alerts_emitted"] or 1),
                "per_modality_recall": per["incident_recall"],
                "entity_agnostic_recall": ent["incident_recall"],
            }
        )
    return {"experiment": "E5.4_cross_modal_contribution", "rows": rows}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {
        "paper": "P5 Cross-modal detection fusion and multi-analytics alert deduplication",
        "results": [e5_1_baselines(), e5_2_window_sweep(), e5_3_derived_window(), e5_4_cross_modal()],
    }
    path = RESULTS / "p5_fusion.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    for block in out["results"]:
        print("\n#", block["experiment"])
        for row in block["rows"]:
            print(" ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
