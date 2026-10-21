from app.evidence.grounding import verify

SHEET = """BEST: [E1] Lakeside Garden, Thu 22 Oct 17:00–18:00, score [E2] 81/100
FACTS:
[E3] feels-like temperature 29.5 °C (17:00–18:00)
[E4] rain probability up to 10 % (17:00–18:00)
[E5] US AQI up to 72 (17:00–18:00)
[E6] 12 min / 4.1 km by road (free-flow)
[E7] sunrise 06:31, sunset 18:05 on Thu 22 Oct
CARRY: about 0.7 L of water per person"""
IDS = {f"E{i}" for i in range(1, 8)}
NAMES = ["Lakeside Garden", "Example Nagar"]


def test_grounded_answer_passes():
    a = ("**Best pick:** Lakeside Garden [E1], Thu 22 Oct 17:00–18:00 [E2].\n- feels like 29.5 °C [E3], rain 10% [E4]\n"
         "- US AQI 72 [E5]; 12 min drive [E6]; sunset 18:05 [E7]. Carry 0.7 L water.")
    r = verify(a, SHEET, IDS, NAMES, "Lakeside Garden")
    assert r.status == "passed", r.violations


def test_invented_number_fails():
    r = verify("Lakeside Garden [E1] will be 24 °C [E3].", SHEET, IDS, NAMES, "Lakeside Garden")
    assert any("24" in v for v in r.violations)


def test_number_with_wrong_unit_fails():
    r = verify("Lakeside Garden [E1]: carry 12 L of water [E6].", SHEET, IDS, NAMES)
    assert any("12 L" in v for v in r.violations)


def test_invented_place_fails():
    r = verify("Try Marine Drive Promenade [E1] at 17:00 [E2].", SHEET, IDS, NAMES)
    assert any("Marine Drive" in v for v in r.violations)


def test_unknown_citation_and_time():
    r = verify("Lakeside Garden [E1] at 07:00 [E99].", SHEET, IDS, NAMES)
    assert any("E99" in v for v in r.violations) and any("07:00" in v for v in r.violations)


def test_not_recommended_framing_and_polarity():
    sheet = SHEET + "\nAVOID / TIMING:\n[E8] Thu 08:00–10:00: US AQI 171 is above the 100 limit"
    a = "**Best pick:** Lakeside Garden [E1] with good air quality [E5]."
    r = verify(a, sheet, IDS | {"E8"}, NAMES, "Lakeside Garden", recommended=False)
    joined = " ".join(r.violations)
    assert "best pick" in joined and "good air" in joined and "doesn't say so" in joined


def test_citation_must_sit_next_to_its_value():
    sheet = ("BEST: [E1] Course A, Sat 24 Oct 06:00–10:00, score [E2] 73.8/100\n"
             "OTHER OPTIONS:\n[E3] Course B, Sat 24 Oct 06:00–10:00, score [E4] 71.7")
    vals = {"E1": "Course A", "E2": 73.8, "E3": "Course B", "E4": 71.7}
    bad = "Course A [E1] scores 73.8 [E2]; Course B [E3] scores [E2] 71.7."
    good = "Course A [E1] scores 73.8 [E2]; Course B [E3] scores 71.7 [E4]."
    assert any("[E2] cited next to 71.7" in v for v in verify(bad, sheet, set(vals), ["Course A", "Course B"],
                                                               ledger_values=vals).violations)
    assert verify(good, sheet, set(vals), ["Course A", "Course B"], "Course A", ledger_values=vals).status == "passed"


def test_soft_positive_air_words_are_caught():
    sheet = SHEET + "\nCARRY: an N95 mask for the journey (US AQI up to 175)"
    r = verify("Lakeside Garden [E1]: air quality is acceptable [E5].", sheet, IDS, NAMES)
    assert any("acceptable" in v for v in r.violations)


def test_looping_answer_fails():
    loop = "Lakeside Garden [E1], Thu 22 Oct 17:00–18:00, score 81/100; " * 3
    r = verify("**Best pick:** " + loop, SHEET, IDS, NAMES, "Lakeside Garden")
    assert any("repeated text" in v for v in r.violations)


def test_health_warning_must_be_stated_and_not_softened():
    sheet = SHEET.replace("FACTS:", "HEALTH WARNING: [E8] US AQI 177 is unhealthy (above 150): keep exertion light and short\nFACTS:")
    ids = IDS | {"E8"}
    soft = "**Best pick:** Lakeside Garden [E1]; air is manageable [E5]."
    r = verify(soft, sheet, ids, NAMES, "Lakeside Garden")
    assert any("health warning" in v for v in r.violations) and any("manageable" in v for v in r.violations)


def test_false_absence_claim_fails():
    sheet = SHEET + "\nOTHER OPTIONS:\n[E8] Hill Park, Thu 22 Oct 17:00–18:00, score [E9] 70"
    r = verify("**Best pick:** Lakeside Garden [E1]. No other options available.", sheet, IDS | {"E8", "E9"}, NAMES)
    assert any("other options" in v for v in r.violations)
