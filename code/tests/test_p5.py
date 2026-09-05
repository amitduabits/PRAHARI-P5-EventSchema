import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from prresearch.p5_fusion.collapse import EntityAgnosticCollapse, NaiveOr, PerModalityDedup, score
from prresearch.p5_fusion.experiment import synth_detections
from prresearch.p5_fusion.schema import optimal_window_s


def test_geometric_window_shrinks_with_speed():
    assert optimal_window_s(60, 40) > optimal_window_s(60, 100)
    assert optimal_window_s(120, 60) > optimal_window_s(60, 60)


def test_collapse_reduces_alerts_without_losing_incidents():
    events, _, _ = synth_detections(1500, 0.45, "t5")
    naive = score(NaiveOr().run(events), 1500)
    ent = score(EntityAgnosticCollapse(120.0).run(events), 1500)
    assert ent["alerts_emitted"] < naive["alerts_emitted"] * 0.4
    assert ent["incident_recall"] == 1.0


def test_entity_agnostic_beats_per_modality_when_dual_tagged():
    events, _, _ = synth_detections(1500, 1.0, "t5:dual")
    per = score(PerModalityDedup(120.0).run(events), 1500)
    ent = score(EntityAgnosticCollapse(120.0).run(events), 1500)
    assert ent["alerts_emitted"] < per["alerts_emitted"]


def test_long_window_masks_distinct_incidents():
    events, _, _ = synth_detections(1500, 0.45, "t5")
    short = score(EntityAgnosticCollapse(15.0).run(events), 1500)
    long_ = score(EntityAgnosticCollapse(1800.0).run(events), 1500)
    assert long_["incidents_masked_by_over_collapse"] > short["incidents_masked_by_over_collapse"]
    assert long_["distinct_incident_recall"] < short["distinct_incident_recall"]


def test_region_occupancy_uses_the_same_record_shape():
    events, _, _ = synth_detections(800, 0.45, "t5")
    regions = [e for e in events if e.entity_type == "region"]
    assert regions
    assert all(e.modality == "occupancy" for e in regions)
    assert set(EntityAgnosticCollapse(120.0).run(regions)[0].members[0].__dataclass_fields__) == set(
        events[0].__dataclass_fields__
    )
