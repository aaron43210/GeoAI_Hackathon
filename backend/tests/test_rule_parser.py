import pytest

from app.agent.rule_parser import parse


@pytest.mark.parametrize("q,activity,location,profiles", [
    ("where should i jog tmrw morning around juhu", "running", "juhu", []),
    ("My dad is 72, is it fine for him to take a walk near Shivaji Park this evening?", "walking", "Shivaji Park", ["senior"]),
    ("kal shaam football khelna hai kharghar mein", "football", "kharghar", []),
    ("good trek near pune for sunday, i have asthma", "hiking", "pune", ["asthma"]),
    ("need tennis courts within 5 km of Indiranagar tomorrow 7am", "tennis", "Indiranagar", []),
    ("Is it okay to cycle to work at 8 AM?", "cycling", None, []),
    ("picnic spots near lalbagh with my 70 year old mother", "picnic", "lalbagh", ["senior"]),
])
def test_parse(q, activity, location, profiles):
    i = parse(q)
    assert i.activity == activity
    assert (i.location_text or None) == location
    assert sorted(i.profiles) == sorted(profiles)


def test_radius_and_out_of_scope():
    assert parse("golf within 40 km of Andheri").radius_km == 40
    assert parse("suggest an indoor gym").in_scope is False


def test_unknown_activity_is_not_defaulted():
    assert parse("what should I do outside near Powai").activity is None


def test_time_words_are_not_a_location():
    from app.schemas.intent import Intent
    assert Intent(activity="running", location_text="right now").normalised().location_text is None
    assert parse("Is it a good idea to go jogging right now?").location_text is None


def test_shorthand_and_trailing_place():
    from app.agent.rule_parser import tail_place_candidates
    assert parse("picnic tmrw w/ family nr cubbon prk").location_text == "cubbon park"
    assert parse("Is it okay to cycle to work at 8 AM from Wakad?").location_text == "Wakad"
    assert tail_place_candidates("football 5 a side kal shaam dwarka delhi")[0] == "dwarka delhi"
    assert parse("recommend a good restaurant in Bandra").in_scope is False


def test_model_output_with_invalid_fields_keeps_the_rest():
    from app.schemas.intent import Intent
    i = Intent.from_model_output({"activity": "cycling", "location_text": "Gurugram",
                                  "preferences": ["wind", "rain", "quiet"], "radius_km": "far"})
    assert i.activity == "cycling" and i.location_text == "Gurugram" and i.preferences == ["quiet"] and i.radius_km is None


def test_location_terminators_and_scope_cues():
    from app.agent.rule_parser import has_in_scope_cue
    assert parse("Best cycling routes this Sunday morning near Gurugram considering wind, rain").location_text == "Gurugram"
    assert has_in_scope_cue("Compare Gurugram, Noida and Delhi for outdoor activities this Saturday")
