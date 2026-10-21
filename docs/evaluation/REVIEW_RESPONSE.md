# Response to the external API review

Live endpoint: https://aaronr43210-terrapulse-ai.hf.space · code: this repository · observed results in this folder.

## 1. Safety is decided by code before any text is generated

The reviewer recommended enforcing safety before generation rather than correcting text afterwards. That is now the
architecture end to end:

1. **Assessment**: live weather, air quality and map data become per-venue, per-hour values (`scoring/grid.py`).
2. **Policy**: three-valued rules in `knowledge/rules.yaml` decide *avoid* / *caution* / *ok* for every hour
   (a missing value is *unknown*, never "fine").
3. **Scoring under constraints**: hours that hit an *avoid* rule can't be recommended. A *high-severity caution*
   (e.g. US AQI > 150) caps the score at 55 (`knowledge/scoring.yaml`), and ties under the cap are broken by the
   uncapped quality, so comfort can't hide a health risk.
4. **The decision line is rendered by code**, not the model: best pick or *Not recommended*, time window, score and
   any **health warning** (`advice/template.py: headline`). The model only writes the explanation underneath, and any
   headline it writes itself is discarded.
5. **Verifier** (`evidence/grounding.py`): every number with its unit, time, place, citation and citation-to-value
   pairing; repeated text; false "no other options"; **comparisons** ("45 exceeds 200" is rejected); and
   **severity in both directions**: no reassurance about a flagged value ("manageable", "air quality is good") and
   no alarm about an unflagged one ("AQI 45 exceeds the safe limit").
6. **Deterministic fallback**: if generation or verification fails (including GPU quota), the answer is rendered
   from the fact sheet. The API returns 200 with grounded content, never a 500.

## 2. The reviewer's 12 scenarios → automated tests (all offline, upstream APIs mocked)

| # | Scenario | Test |
|---|---|---|
| 1 | Unhealthy AQI during golf: explicit warning, no reassurance | `test_engine::test_high_severity_caution_caps_score`, `test_grounding::test_health_warning_must_be_stated_and_not_softened` |
| 2 | More severe pollution for a sensitive runner | `test_engine::test_asthma_gates_aqi_that_is_fine_for_others`, `test_engine::test_profiles_only_tighten` |
| 3 | Good weather can't override the health policy | `test_review_scenarios::test_model_cannot_overwrite_the_decision`, score cap test |
| 4 | Missing AQI disclosed, not assumed clean | `test_review_scenarios::test_missing_air_quality_is_disclosed_not_assumed_clean`, `test_engine::test_three_valued_logic` |
| 5 | GPU quota exhausted → valid fallback, not HTTP 500 | `test_pipeline::test_gpu_quota_exhausted_still_answers` |
| 6 | Model repeats itself | `test_grounding::test_looping_answer_fails` |
| 7 | `access=no` venue excluded | `test_review_scenarios::test_access_no_venue_is_excluded` (this found a real bug: YAML read `no` as `false`) |
| 8 | Opening hours unavailable → never claimed open | `test_review_scenarios::test_opening_hours_not_claimed_and_alternatives_are_eligible` |
| 9 | Missing hourly forecast → no unsupported time claims | `test_pipeline::test_weather_down_refuses_to_recommend` (no forecast, no time recommendation) |
| 10 | Alternatives independently checked | same test as 8 (every alternative must be non-*avoid*) |
| 11 | Multi-city request | explained limitation: one area per request; the assistant asks for the activity/area (`review_queries_live.json`) |
| 12 | Conflicting model output preserves the decision | `test_review_scenarios::test_model_cannot_overwrite_the_decision` |

Plus `test_logic_regressions.py`, which covers logic bugs found in our own audit: golden-hour overlap, "this/next
weekend" asked at the weekend, industrial exposure measured from the nearest edge, ranking under score caps, "a 5 km
run" not read as a search radius, failed geocodes not cached for a month. The suite has 71 tests, all passing.

## 3. Observed live results (`review_queries_live.json`)

| Query | Status | Pick | Score | Answer | Latency |
|---|---|---|---|---|---|
| Golf near Gurugram | partial (map context unavailable) | DLF Golf and Country Club, Sat 06:00–10:00 | 55 (capped, caution: AQI 177) | model, verified | 29.9 s |
| 5 km run, weather + AQI | partial | Tikona Park, 06:00–07:00 | 53.1 caution | template | 31.4 s |
| Cycling, wind/rain/AQI | partial | Tikona Park, Sun 06:00–07:00 | 55 caution | template | 23.9 s |
| Compare three cities | clarification | asks which activity (one area per request) | – | – | 2.4 s |
| Football, rain/wind/temp | partial | Nehru Stadium, 06:00–07:00 | 55 caution | template | 19.9 s |
| Safest activity, 7 days | clarification | asks which activity (safety depends on it) | – | – | 2.7 s |

**Metrics, stated honestly:** across these runs 1 of 4 recommendations used the model's prose and 3 used the
verified template. The stricter checks (health-warning wording, comparisons, alarm/reassurance polarity) reject
more small-model drafts; the decision and facts are identical either way. Typical latency is 20–30 s, dominated by
OpenStreetMap Overpass being unavailable during testing (Nominatim fallback), and ~2–3 s for clarifications.
A multi-turn conversation transcript (greeting → plan → "why?" → "what about…" → profile change → thanks) is in
`chat_transcript_live.txt`.

## 4. Known limitations
- Small-model phrasing: model prose is often rejected by the verifier, so many answers are templates. Correct but plainer.
- OSM opening hours, when present, may be outdated; bookings and tee times are never checked (stated in every answer).
- One area per request; comparing several cities in one question is not supported.
- AQI comes from a ~40 km CAMS grid; differences between nearby venues come from the modelled wind × road/industry exposure.
- The free ZeroGPU host meters GPU runs per caller; an `Authorization: Bearer <HF token>` header gives the caller's own quota.
