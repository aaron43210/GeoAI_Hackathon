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


def test_known_terms_are_not_places():
    r = verify("Lakeside Garden [E1]: the Air Quality Index is 72 [E5].", SHEET, IDS, NAMES)
    assert not any("Air Quality Index" in v for v in r.violations)


def test_inverted_comparison_is_rejected():
    sheet = SHEET + "\nLIMITS USED: US AQI ≤ 200"
    bad = verify("Lakeside Garden [E1]: US AQI 72 [E5], exceeding the acceptable limit of 200.", sheet, IDS, NAMES)
    assert any("false comparison" in v for v in bad.violations)
    good = verify("Lakeside Garden [E1]: US AQI 72 [E5], well below the limit of 200.", sheet, IDS, NAMES)
    assert not any("false comparison" in v for v in good.violations)


def test_alarm_about_unflagged_air_quality_is_rejected():
    bad = verify("Lakeside Garden [E1]: the US AQI exceeds the safe limit and is harmful [E5].", SHEET, IDS, NAMES)
    assert any("does not flag" in v for v in bad.violations)
    flagged_sheet = SHEET + "\nCARRY: an N95 mask for the journey (US AQI up to 175)"
    ok = verify("Lakeside Garden [E1]: unhealthy air today [E5].", flagged_sheet, IDS, NAMES)
    assert not any("does not flag" in v for v in ok.violations)


def test_false_time_comparison_is_rejected():
    sheet = SHEET.replace("sunset 18:05", "sunset 17:49")
    bad = verify("Lakeside Garden [E1]: play 19:00–20:00, well before sunset at 17:49 [E7].", sheet, IDS, NAMES)
    assert any("false time comparison" in v for v in bad.violations)
    ok = verify("Lakeside Garden [E1]: play 17:00–18:00, before sunset at 18:05 [E7].", SHEET, IDS, NAMES)
    assert not any("false time comparison" in v for v in ok.violations)


def test_within_safe_limits_is_softening_when_flagged():
    sheet = SHEET + "\n[E8] caution: US AQI 129 (above 100)"
    r = verify("Lakeside Garden [E1]: US AQI 129 [E8] is still within safe limits.", sheet, IDS | {"E8"}, NAMES)
    assert any("within safe limits" in v for v in r.violations)


FLAGGED = """BEST: [E1] Lakeside Garden, Fri 23 Oct 19:00–20:00, score [E2] 62/100
FACTS:
[E5] US AQI up to 135 (19:00–20:00)
[E9] caution: US AQI 135 is above 100 (unhealthy for sensitive groups); carry an N95"""
FIDS = {"E1", "E2", "E5", "E9"}


def test_wrong_aqi_band_and_downplaying_are_rejected():
    # the sentence round-2 wrote live on 23 Oct
    a = ("Lakeside Garden [E1] suits an evening ride, and the air quality, while moderate at a US AQI of 135 [E5], "
         "won't pose a significant risk for sensitive groups.")
    v = verify(a, FLAGGED, FIDS, NAMES, "Lakeside Garden").violations
    assert any("'moderate'" in x for x in v) and any("won't pose" in x for x in v), v


def test_correct_aqi_band_passes():
    a = ("Lakeside Garden [E1] works, but the US AQI of 135 [E5] is unhealthy for sensitive groups, "
         "so keep it light and wear an N95 [E9].")
    r = verify(a, FLAGGED, FIDS, NAMES, "Lakeside Garden")
    assert r.status == "passed", r.violations
    assert verify("The US AQI of 72 [E5] at Lakeside Garden is moderate.", SHEET, IDS, NAMES).status == "passed"


def test_wrong_feels_like_band_is_rejected():
    sheet = SHEET.replace("29.5 °C", "31 °C")
    a = "Lakeside Garden [E1] offers some relief with cool weather (feels like about 31 °C [E3])."
    assert any("'cool'" in v for v in verify(a, sheet, IDS, NAMES, "Lakeside Garden").violations)
    ok = "Lakeside Garden [E1] will be warm, feels like about 31 °C [E3]."
    assert verify(ok, sheet, IDS, NAMES, "Lakeside Garden").status == "passed"
