"""Rules engine (three-valued), profile tightening, wind-aware exposure."""
from app.geospatial.context import SourcePoint
from app.geospatial.exposure import exposure
from app.knowledge import get_knowledge
from app.scoring.rules_engine import Tri, evaluate_condition, evaluate_rules


def test_three_valued_logic():
    cond = {"var": "us_aqi", "op": ">", "limit": "us_aqi_max"}
    assert evaluate_condition(cond, {"us_aqi": 180}, {"us_aqi_max": 150})[0] == Tri.TRUE
    assert evaluate_condition(cond, {"us_aqi": 90}, {"us_aqi_max": 150})[0] == Tri.FALSE
    assert evaluate_condition(cond, {}, {"us_aqi_max": 150})[0] == Tri.UNKNOWN
    assert evaluate_condition(cond, {"us_aqi": 999}, {})[0] == Tri.FALSE   # limit not defined → not applicable


def test_avoid_suppresses_matching_caution():
    kb = get_knowledge()
    hits = evaluate_rules(kb.rules, {"apparent_temperature": 45}, kb.effective_limits("running", []))
    ids = {h.id for h in hits}
    assert "GEN-HEAT" in ids and "GEN-HEAT-CAUTION" not in ids


def test_profiles_only_tighten():
    kb = get_knowledge()
    for act in kb.activities:
        base = kb.effective_limits(act, [])
        for pid in kb.profiles:
            tight = kb.effective_limits(act, [pid])
            for k, v in base.items():
                if k.endswith("_min"):
                    assert tight[k] >= v
                else:
                    assert tight[k] <= v


def test_asthma_gates_aqi_that_is_fine_for_others():
    kb = get_knowledge()
    v = {"us_aqi": 160}      # EPA: sensitive groups avoid prolonged exertion above 150, others above 200
    assert not [h for h in evaluate_rules(kb.rules, v, kb.effective_limits("walking", [])) if h.decision == "avoid" and h.state == Tri.TRUE]
    assert [h for h in evaluate_rules(kb.rules, v, kb.effective_limits("walking", ["asthma"]))
            if h.decision == "avoid" and h.state == Tri.TRUE]


def test_wind_shift_changes_exposure():
    # an expressway 200 m to the EAST of the venue (x = +200 m)
    road = [SourcePoint(200, y, 0.2, "road", "Expressway") for y in range(-300, 301, 50)]
    from_east = exposure(road, wind_from_deg=90, wind_kmh=12, traffic_factor=1.0)
    from_west = exposure(road, wind_from_deg=270, wind_kmh=12, traffic_factor=1.0)
    assert from_east.index > 0.1 and from_west.index == 0
    assert from_east.dominant_source == "Expressway"


def test_calm_wind_counts_all_directions():
    road = [SourcePoint(200, 0, 0.5, "road", "Road")]
    assert exposure(road, wind_from_deg=270, wind_kmh=2, traffic_factor=1.0).index > 0


def test_missing_wind_is_unknown_not_zero():
    assert exposure([SourcePoint(100, 0, 1, "road", "R")], None, None, 1.0) is None


def test_every_activity_is_complete():
    kb = get_knowledge()
    assert len(kb.activities) >= 10
    for a in kb.activities.values():
        assert a.venues and a.synonyms and a.nominatim_phrases
        assert abs(sum(a.weights.values()) - 1.0) < 0.05, a.id


def test_high_severity_caution_caps_score():
    from app.scoring.grid import _capped
    from app.scoring.rules_engine import evaluate_rules
    kb = get_knowledge()
    hits = evaluate_rules(kb.rules, {"us_aqi": 177}, kb.effective_limits("golf", []))
    assert any(h.id == "GEN-AQI-UNHEALTHY" for h in hits)
    assert _capped(90.0, hits) == kb.scoring["severity_score_cap"]["high"]


def test_knowledge_word_lists_are_strings():
    """YAML 1.1 turns bare on/off/yes/no into booleans; every word list must stay strings."""
    import glob
    import yaml
    for f in glob.glob(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app/knowledge/*.yaml")):
        def walk(x):
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    assert not isinstance(v, bool), f"boolean in a list in {f}: quote it"
                    walk(v)
        walk(yaml.safe_load(open(f)))
