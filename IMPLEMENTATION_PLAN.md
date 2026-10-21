# TerraPulse AI — Implementation Plan

GeoAI Hackathon @ CSRE, IIT Bombay · Deadline **Fri 24 Oct 2026, 23:59 IST** · Plan written Tue 21 Oct 2026

Companion docs: [CONNECTIONS.md](CONNECTIONS.md) (how every piece connects) · [AGENTS.md](AGENTS.md) (rules for anyone/any AI agent editing the repo) · [MEMORY.md](MEMORY.md) (decision log + progress).

> ## ✅ Status at end of 21 Oct (read this first)
> Day-1 and most of Day-2 scope is **built and live-tested locally**: knowledge YAML, live-only data layer (Overpass
> race + breaker + Nominatim fallback), time resolver, venue × hour grid with wind-aware exposure, three-valued rules,
> advice, evidence ledger, grounding verifier (numbers + units, times, names, citation adjacency, severity wording),
> SLM runtime (Qwen3-1.7B GGUF), `/ask` + SSE + `/recommend` `/explain` `/venues`, new React UI, 45 offline tests, live
> eval harness, Colab notebooks, Dockerfile, README.
> Changes from the original plan: **model chosen by measured bake-off** (Qwen3-1.7B beat Qwen3.5-2B; see README);
> **Geofabrik fallback dropped** (no preloaded data) → Nominatim category search; AQI limits aligned with US EPA
> activity guidance; single container on port 7860; training **only in Colab**.
> **Remaining:** git + GitHub → `upload_hf_model.py --mode base` → Colab notebooks 01/02 → deploy + uptime monitor →
> slides → final eval and demo rehearsal. See MEMORY.md "Progress".

---

## 0. What the judges will check (requirements → how we satisfy them)

| # | Requirement (from problem statement) | How we satisfy it | Where |
|---|---|---|---|
| R1 | Parse activity, area/radius, time range, personal details (age, health) from free-form text | SLM intent extraction (JSON-schema constrained) + deterministic time resolver | `slm/`, `agent/understand.py`, `agent/time_resolver.py` |
| R2 | Consider **several places across upcoming hours** | Venue × hour score grid (≤8 venues × all hours in range, up to 5 days) | `scoring/grid.py` |
| R3 | Geospatial context: wind direction, roads & traffic, industry, water, green cover | One combined Overpass fetch → shapely features → **per-hour wind-aware exposure** | `geospatial/` |
| R4 | Clear recommendation + what to carry, what to watch for, when to avoid | Deterministic advice rules with evidence IDs, phrased by the SLM | `advice/rules.py`, `slm/composer.py` |
| R5 | Every claim backed by live data, **no invented weather or places** | Evidence ledger + grounding verifier; no synthetic fallbacks anywhere | `evidence/` |
| R6 | Plain text or rich output (maps, images) | `/ask` returns markdown + JSON + GeoJSON; React map/heatmap UI; `?format=text` for curl | `api/ask.py`, `frontend/` |
| R7 | ≥10 activities; personal profiles tighten limits | 10 required + walking, yoga, birdwatching; profile modifiers (asthma, senior, child, pregnancy, cardiac, heat-sensitive) | `scoring/activity_profiles.py` |
| R8 | Messy queries: missing location, vague time, out of scope → handle gracefully, **don't guess** | Clarification + scope gate; no default city | `agent/clarify.py` |
| R9 | Open SLM **≤3B params**, self-run, on HF Hub, `from_pretrained` loads, model card, open licence | Qwen3-1.7B (Apache-2.0) + LoRA, merged + adapter + GGUF repos | `training/`, HF |
| R10 | Live HTTP endpoint for the whole judging window | Docker deploy + uptime monitor + hot standby | `deploy/` |
| R11 | README, 4–6 slides, disclosures (models, data, AI coding tools) | Section 11 | root |
| R12 | No hosted LLMs, no paid APIs, no hard-coded answers, no answering from memory | Enforced by AGENTS.md rules + verifier + eval set that differs from examples | everywhere |

---

## 1. Audit of the current codebase (done 21 Oct)

All 12 existing tests pass, but they don't test grounding. A live run of two example queries showed the current code **breaks R5 and R8**, which are core rules. Everything below is verified, not guessed.

### 🔴 Blockers (must fix first — these would get us disqualified or embarrassed live)

| # | Finding | File | Evidence |
|---|---|---|---|
| B1 | **Fabricated venues** when Overpass fails ("Central Botanical Park Loop", "Emerald Greenway Veloway"), still labelled as coming from *OpenStreetMap & Overpass API* | `data_services/osm_service.py` `_generate_synthetic_venues` | live run returned them |
| B2 | **Fabricated weather** (`_generate_synthetic_weather`, sine-wave diurnal curve) | `data_services/weather_service.py` | code |
| B3 | **AQI endpoint path wrong**: `/air_quality` → 404 every time → hard-coded "baseline" PM2.5 16.5 is served as if real | `data_services/air_quality_service.py` | live 404; correct path is `/v1/air-quality` (verified 200) |
| B4 | **Geocoding falls back to Bengaluru coordinates** for any unknown place; Powai (Mumbai) was placed in Bengaluru | `data_services/geocoding_service.py` | live run: lat 12.99 |
| B5 | Missing location silently → "Bengaluru default"; unknown activity → "Running default" | `agent/intent_parser.py` | live run |
| B6 | Fake geospatial numbers: green cover = `len(elements) × 12000 m²`, floor 0.25; fixed baselines (road 420 m, industry 2800 m, green 0.68) when Overpass fails | `geospatial/*.py` | code |
| B7 | Industrial "downwind" check always uses **wind = 180°** (never the forecast) | `geospatial/industrial_analysis.py` + orchestrator call | code |
| B8 | Default model `microsoft/phi-3-mini-4k-instruct` is **3.8B > 3B limit** | `config.py`, `.env.example`, README | model card |
| B9 | Frontend serves a **hard-coded fake recommendation** (Osho Teerth, 21.4 °C, PM2.5 12) whenever the backend errors | `frontend/src/services/api.ts` `getFallbackRecommendation` | code |
| B10 | Evidence records state invented methods/reliabilities ("Bilinear interpolation from 0.1° grid", reliability 0.92) | `evidence/provenance.py` | code |
| B11 | `/explain` returns hard-coded rationale text ("Buffer zones ensure separation…") | `api/explain.py` | code |

### 🟠 Functional gaps vs. the problem statement

- Location regex captured `"Powai In The Next Two Days"`; "walk" not recognised (→ running).
- Only **one date**, and only **current** AQI (not hourly). "Next two days", "Saturday or Sunday", "this weekend" can't be compared.
- No water-body analysis, no power lines, no daylight/sunset logic (photography!), no rush-hour travel, no "when to avoid", no "what to carry".
- No age / child / senior profile ("my parents", "kids' cricket").
- `date.today()` uses **server timezone** — on a UTC host, "tomorrow" is wrong between 00:00–05:30 IST.
- N venues × 3 Overpass calls each (+1) → rate-limited quickly; Overpass is the flakiest dependency (21 Oct probe: overpass-api.de and private.coffee unreachable, maps.mail.ru 200 in 22 s).
- `api/venues.py` builds its own cache/services at import time (not the shared orchestrator ones).
- Frontend calls `localhost:8080`, backend runs on `8000`.
- `datetime.utcnow()` deprecated (107 warnings) — use tz-aware datetimes.

### ✅ Keep (good foundations)

FastAPI app + lifespan, Pydantic v2 schemas, async httpx services, cache abstraction, the `SafetyConstraints` idea (deterministic hard gates), `RankingEngine` weights idea, evidence model, React+Vite+Leaflet frontend shell, Docker files, tests layout.

---

## 2. Target architecture

**Core design choice (and how we defend it to judges):** the SLM does the *language* work: understanding messy, Hinglish or vague questions, and writing the advice. Arithmetic, safety gates and ranking are **deterministic and auditable**. A 1.7B model is good at language and bad at arithmetic and calendars, so we give each side the job it does well. The rules say *"answering from the model's memory instead of the data it fetched"* is not allowed. Our verifier makes that impossible: any number or place name in the answer that isn't in the fetched evidence gets rejected.

```
                      POST /api/v1/ask  {question, session_id?, user_location?}
                                   │
  ┌────────────────────────────────▼────────────────────────────────┐
  │ 1 UNDERSTAND  SLM call #1 (JSON-schema constrained, ~120 tok)    │
  │   → activity, place text, radius, time phrase (verbatim),        │
  │     profile (age/asthma/…/children/seniors), prefs, scope flag   │
  │   + TimeResolver (deterministic, Asia/Kolkata): phrase → windows │
  │   + Clarify/Scope gate → may return early with a question        │
  └────────────────────────────────┬────────────────────────────────┘
  ┌────────────────────────────────▼────────────────────────────────┐
  │ 2 LOCATE  Nominatim (countrycodes=in) → point + bbox; ambiguity  │
  │           → ask; browser geolocation if "near me"                │
  └────────────────────────────────┬────────────────────────────────┘
  ┌────────────────────────────────▼────────────────────────────────┐
  │ 3 FETCH (parallel, area-level, cached, rate-limited)             │
  │   Overpass (1 combined query, mirror race) ─ venues + roads +    │
  │       industry + water + green + power lines                     │
  │   Open-Meteo forecast (all venues in ONE multi-coord call)       │
  │   Open-Meteo air-quality hourly (multi-coord)                    │
  │   OSRM /table (origin → all venues, 1 call)                      │
  │   WAQI nearest station (observed now)  · OpenTopoData (hiking)   │
  └────────────────────────────────┬────────────────────────────────┘
  ┌────────────────────────────────▼────────────────────────────────┐
  │ 4 ANALYSE  shapely features per venue (static) +                 │
  │            per-hour wind-aware exposure (dynamic) + daylight     │
  │ 5 SCORE    venue × hour grid: hard gates → soft score            │
  │            → best (venue, window), alternatives, avoid-windows   │
  │ 6 ADVISE   rules → carry / watch-for / avoid, each with ev IDs   │
  └────────────────────────────────┬────────────────────────────────┘
  ┌────────────────────────────────▼────────────────────────────────┐
  │ 7 COMPOSE  SLM call #2: compact fact sheet (E1…En) → markdown    │
  │            answer citing [E#]                                    │
  │ 8 VERIFY   every number/place/time ∈ fact sheet? citations real? │
  │            fail → 1 repair try → deterministic template          │
  └────────────────────────────────┬────────────────────────────────┘
                                   ▼
      {answer_markdown, recommendation, alternatives, avoid, advice,
       grid, geojson, evidence[], sources[], grounding, timings, model}
```

Streaming variant `GET /api/v1/ask/stream` (SSE) emits stage events ("Found 14 parks near Powai…", "Scoring 6 venues × 38 hours…") and then answer tokens. This hides latency and makes the demo feel interactive.

---

## 3. Model plan

### 3.1 Choice

| Candidate | Params | Licence | Verdict |
|---|---|---|---|
| **Qwen3-1.7B** (`Qwen/Qwen3-1.7B`) | 1.7B | Apache-2.0 | **Primary.** Clearly under 3B, open licence, strong JSON/tool use, multilingual (Hinglish, Indian place names); thinking can be turned off (`enable_thinking=False`); Unsloth + llama.cpp support |
| SmolLM3-3B (`HuggingFaceTB/SmolLM3-3B`) | ~3.08B | Apache-2.0 | Backup. Strong at 3B, but 3.08B is borderline on a strict "≤3B" reading |
| Llama-3.2-3B-Instruct | 3.21B | Llama Community (not OSI) | Avoid: over 3B *and* the licence is debatable |
| Qwen2.5-3B-Instruct | 3.09B | Qwen Research (non-commercial) | Avoid: licence doesn't count as open (Qwen2.5-1.5B is Apache-2.0, a fallback if needed) |
| Phi-3-mini / Phi-4-mini | 3.8B | MIT | **Disallowed (>3B)**. The current repo uses it, so change it |
| Gemma-2-2B-it | 2.6B | Gemma Terms | OK in size, custom licence, weaker at JSON |

> Day-1 bake-off (1.5 h): run the 30-query intent set + 10 compose fact sheets through Qwen3-1.7B, SmolLM3-3B and any newer Apache-2.0 model ≤3B released since. Pick on (a) JSON slot accuracy, (b) groundedness pass rate, (c) CPU latency. If nothing beats Qwen3-1.7B by more than 5 points, keep it.

### 3.2 Two jobs, one model (one adapter trained on both tasks)

1. **Intent extraction**: system prompt + question (+ previous intent for follow-ups) → JSON matching `IntentSchema` (Section 5.1). Constrained decoding (llama.cpp JSON schema/GBNF) means invalid JSON can't happen. The model copies the time phrase **verbatim**; it never does calendar arithmetic.
2. **Answer composition**: fact sheet → ≤220-word markdown with `[E#]` citations, fixed section order (Best pick, Why, Carry, Watch for, Avoid, Alternatives, Data note).

### 3.3 Fine-tuning (Colab via the VS Code Colab extension, free T4)

Fine-tuning helps **but the system doesn't depend on it**. Base Qwen3-1.7B plus prompts must already work by Day 2; the LoRA adapter is the Day-3 upgrade.

**Dataset A, intent (~3,000 examples, labels exact by construction)** — `training/gen_intent_data.py`
- Generator over: 13 activities × ~40 synonyms each ("jog", "5-a-side", "nets practice", "trek", "shaadi/wedding", "golden hour shoot", "morning walk") × ~300 Indian localities/cities × time phrases (today/tonight/tomorrow evening/this weekend/Sat or Sun/next two days/after work/at 8 AM/kal subah/parso shaam/sunset) × profiles (mild asthma, 68-year-old, my parents, kids, 7 months pregnant, heart patient) × radius/travel ("within 15 km", "under 30 min").
- Messiness: typos, lowercase, no punctuation, Hinglish, emojis, missing location, missing time, two activities, out-of-scope ("book a cab", "indoor gym", "stock tips"), unsafe asks.
- Optional paraphrase pass with an **open** teacher (Qwen3-8B on Colab) to diversify wording. Labels stay from the generator. No hosted LLMs, which keeps the disclosure clean.
- Split 90/10. Plus a hand-written **eval set of 80 queries** (`eval/queries.jsonl`) that are *not* the example queries, so we can show we didn't hard-code answers.

**Dataset B, compose (~1,500 examples)** — `training/gen_compose_data.py`
- Inputs are **real fact sheets** from our own pipeline: run ~150 varied queries over ~40 Indian cities across 2–3 days (cache the raw API responses into `training/raw/` so it's reproducible).
- Targets are drafts from the open teacher (Qwen3-8B, 4-bit on T4) given the fact sheet and the style guide. **Keep only drafts that pass our grounding verifier at 100%.**
- Also includes clarification replies, "data unavailable" replies (AQ missing, Overpass down), "it's not safe at any hour → here's why" replies, and out-of-scope replies.

**Training config** (`training/03_finetune_lora.ipynb`, Unsloth)
- Base `Qwen/Qwen3-1.7B`, LoRA r=16, α=32, dropout 0.05, targets q,k,v,o,gate,up,down; max_seq 2048; lr 2e-4 cosine; 2 epochs; effective batch 16; loss on completions only; fp16 on T4. Roughly 45–70 min.
- Evaluate before and after: intent slot exact-match/F1, clarification precision/recall, groundedness pass rate, avg tokens, and CPU latency on the GGUF.

**Publish (3 HF repos, Apache-2.0)**
- `aaronr43210/terrapulse-qwen3-1.7b-lora`: adapter + `adapter_config.json` pointing to `Qwen/Qwen3-1.7B`
- `aaronr43210/terrapulse-qwen3-1.7b`: merged safetensors, so `AutoModelForCausalLM.from_pretrained(...)` works (judges' check)
- `aaronr43210/terrapulse-qwen3-1.7b-GGUF`: Q4_K_M + Q8_0, which is what the server runs
- Model card: base model, what changed (LoRA, data recipe, sizes), intended use and out-of-scope use, eval table before/after, limitations, licence, and the dataset generation scripts linked to the code repo. Optionally push the datasets to an HF dataset repo too.

### 3.4 Serving

- `llama-cpp-python` loads the GGUF once at startup, warms up, and uses `create_chat_completion(response_format={"type":"json_object","schema":…})` for intent.
- Fallback chain: fine-tuned GGUF → base GGUF → (intent) regex parser / (compose) deterministic template. Every response reports which path ran (`model.path_used`) so we never hide a fallback.
- CPU budget (2–4 vCPU): intent ≈ 2–5 s, compose ≈ 8–20 s with a fact sheet of ~600 tokens and output capped at 260 tokens; stream tokens. Target p50 < 20 s, p95 < 45 s end-to-end (cold Overpass is the main variable).

---

## 4. Data sources (all free; disclose all in README)

| Purpose | Source | Call | Key | Limits / etiquette | Cache TTL |
|---|---|---|---|---|---|
| Geocoding | Nominatim | `/search?q=&countrycodes=in&format=jsonv2&limit=5` | none | ≤1 req/s, real UA with contact email | 30 days |
| Venues, roads, industry, water, green, power lines | Overpass API | 1 combined `nwr` query, `out center`/`out geom` | none | race mirrors: overpass-api.de, maps.mail.ru, overpass.kumi.systems, overpass.private.coffee; ≤2 concurrent | 24 h (map features change slowly) |
| OSM fallback | Geofabrik India extracts | `osmium tags-filter` at build → GeoParquet | none | label "OSM extract dated YYYY-MM-DD" | build-time |
| Hourly weather (16 d) + sunrise/sunset | Open-Meteo forecast | multi-coord, `timezone=Asia/Kolkata` | none | 10k/day non-commercial | 30 min |
| Hourly air quality (5 d) | Open-Meteo air-quality (CAMS) | `/v1/air-quality` hourly pm2_5, pm10, no2, o3, us_aqi | none | same | 60 min |
| Observed AQI (ground truth now) | WAQI (CPCB stations) | `/feed/geo:lat;lon/` | free token | 1000/min | 30 min |
| Travel time | OSRM demo | `/table/v1/driving` sources=0 | none | ≤1 req/s, fair use | 6 h |
| Travel fallback | OpenRouteService matrix | `/v2/matrix/driving-car` | free key | 500 matrix/day | 6 h |
| Elevation (hiking) | OpenTopoData | `/v1/srtm30m?locations=…` (≤100 pts) | none | 1 req/s, 1000/day | 30 days |
| Golden hour / twilight | `astral` library (computed from sun position, not fetched) | — | — | — | — |
| Map tiles (UI) | OSM tile server / CARTO basemaps | — | none | attribution "© OpenStreetMap contributors" | browser |
| Stretch | OpenAQ v3, data.gov.in CPCB | REST | free key | — | 30 min |

**Rules:** one shared `httpx.AsyncClient` per host, a per-host token-bucket rate limiter, retry with jitter (2 tries), disk cache (`diskcache`, survives restarts), and **no synthetic fallback ever**. If a source fails, that factor is marked `available: false`, its weight is redistributed, and the answer says so.

**Honest-data notes we state upfront:** CAMS air-quality is a ~40 km global grid, so all venues in a city share the same PM2.5 forecast. Venue-level differences come from our **modelled exposure index** (roads/industry upwind), and we label it as modelled. OSRM has **no live traffic**, so we apply a disclosed rush-hour multiplier.

---

## 5. Backend: detailed design

### 5.1 Intent schema (SLM output)

```json
{
  "in_scope": true,
  "out_of_scope_reason": null,
  "activity": "walking",               // enum of 13 + "unknown"
  "activity_raw": "morning walk",
  "location_text": "Powai",            // null if absent
  "use_user_location": false,          // "near me", "from home", "to work"
  "radius_km": null,                    // "within 15 km" → 15
  "max_travel_min": null,
  "time_text": "morning in the next two days",   // verbatim, resolver handles dates
  "duration_min": null,
  "group": {"children": false, "seniors": false, "size": null},
  "profile": {"age": null, "asthma": "mild", "pregnancy": false, "cardiac": false,
              "heat_sensitive": false, "mobility_limited": false, "notes": null},
  "preferences": {"quiet": false, "shade": false, "water": false, "avoid_traffic": false},
  "compare": null,                      // e.g. ["saturday","sunday"]
  "follow_up_of_previous": false
}
```

### 5.2 TimeResolver (`agent/time_resolver.py`, deterministic, unit-tested heavily)

- "Now" = `datetime.now(ZoneInfo(place_tz or "Asia/Kolkata"))`.
- Lexicon: today, tonight, tomorrow, day after (tomorrow), kal (→ tomorrow, unless past tense), parso, this/next weekend, weekday names, "Saturday or Sunday" (→ compare), "next N days", "after work" (17:30–20:30), "before office" (06:00–09:00), morning 05:30–10:00, afternoon 12:00–16:30, evening 16:30–19:30, night 19:30–22:30, sunrise/sunset (from Open-Meteo daily ±75 min), "at 8 AM" (that hour ±1 h, plus commute duration), subah/dopahar/shaam/raat.
- Output: `List[TimeWindow(day, start, end, label)]` plus a `compare` flag.
- Horizon: AQ 5 days, weather 16 days. Beyond 5 days → "air-quality forecast not available that far out". Beyond 16 → say we can't forecast it and suggest asking closer to the date. Past times → roll forward with a note.
- Missing time → next 24 h of daylight hours, and the answer **says it assumed that**.

### 5.3 Clarify / scope gate (`agent/clarify.py`)

| Situation | Behaviour |
|---|---|
| No location and no `user_location` | Return `status:"clarification_needed"` with a question + quick-reply chips (city list from Nominatim suggestions, "use my location") |
| Ambiguous place (Nominatim top-2 in different cities, e.g. "Andheri" fine, "Model Town" ambiguous) | Ask, showing both options |
| Commute ("cycle to work") without destination | Ask for start and destination (or use the browser location as start). Corridor analysis is a stretch goal; otherwise evaluate the origin area at the departure hour |
| Unknown activity | Ask, listing supported activities; never default to running |
| Out of scope (indoor, non-outdoor, harmful) | Polite decline + what we can do |
| Date beyond horizon | Partial answer + explicit limitation |

### 5.4 Combined geospatial fetch (`data_services/overpass.py`)

One query per (activity group, rounded centre, radius), bounded by `[timeout:25][maxsize:64Mi]`:
- **Candidates** by activity tags (`activity_profiles.py`): e.g. golf `leisure=golf_course`; cricket `sport=cricket`, `leisure=pitch`+`sport~cricket`, `leisure=sports_centre`; tennis `sport=tennis`; football `sport=soccer`; hiking `leisure=nature_reserve`, `route=hiking`, `natural=peak`, `tourism=viewpoint`; photography `tourism=viewpoint`, `natural=beach`, `waterway`/`natural=water` edges, `historic=*`; event/picnic `leisure=park|garden`, `tourism=picnic_site`; running/walking `leisure=park|track`, `highway=footway` promenades.
- **Context** within radius + 3 km: `highway~motorway|trunk|primary|secondary` (geom), `landuse=industrial`, `man_made=works|chimney`, `power=plant|line`, `natural=water`, `waterway=river`, `natural=wood`, `landuse=forest|grass|meadow`, `leisure=park|nature_reserve`.
- Parse into GeoDataFrame-like lists of shapely geometries in a local metric projection (equirectangular around the centre is accurate enough at ≤50 km; no pyproj needed).
- Candidate filtering: must have a `name`, area ≥ activity minimum (cricket ground ≥ 8,000 m², golf ≥ 200,000 m²), dedupe by name and 150 m proximity, skip `access=private|no`. Keep the top 8 by an "activity fit" pre-score (tags, area, distance).

### 5.5 Features (`geospatial/context.py`), static per venue

| Feature | Computation |
|---|---|
| `dist_major_road_m`, `major_road_len_300m` | shapely distance / buffer-intersection length by road class |
| `industrial_within_3km` | list of (distance, bearing, area) |
| `water_within_500m` | nearest distance + name (lake/river/sea) |
| `green_fraction_300m` | **true area** of the green-polygon union ∩ 300 m buffer ÷ buffer area |
| `power_line_within_100m` | bool (matters for kites/drones/events) |
| `lit`, `surface`, `opening_hours`, `access`, `fee` | from OSM tags (show "unknown" if absent) |
| `elevation_m`, `elevation_gain` | OpenTopoData (hiking only) |

### 5.6 Dynamic exposure (`geospatial/exposure.py`), per venue × hour

For each source point *s* (densified road segments every 50 m, industrial polygon samples):
- β = bearing venue→s, θₕ = wind-from direction at hour h, Δ = |θₕ − β| wrapped to 0–180°.
- alignment = cos²(Δ) if Δ < 60° else 0. For wind < 5 km/h (stagnant), alignment = 0.5 in all directions.
- road term = Σ alignment · e^(−d/400 m) · classWeight (motorway 1.0, trunk 0.8, primary 0.6, secondary 0.35) · trafficFactor(h)
- industry term = Σ alignment · e^(−d/1500 m) · areaWeight
- `exposure_index[v,h]` = clip(road + industry, 0, 1). Label it in the UI and answer as "**modelled** exposure (wind × proximity)".

This captures the problem's own example: a ground downwind of the expressway at 8 AM scores worse than the same ground in the evening once the wind shifts.

`trafficFactor(h)` (IST, disclosed heuristic): weekday 08–11 & 17–21 → 1.0; 07, 11–17 → 0.6; night → 0.3; weekend peak ×0.7.

### 5.7 Activity profiles (`scoring/activity_profiles.py`, single source of truth)

One dataclass per activity: OSM tags, min area, default duration, default hours, exertion level (sets AQ sensitivity), comfort band of apparent temperature, gates (max apparent temp, min temp, max gust, max precip prob, max US AQI, needs daylight, thunderstorm = always gate), soft weights, and special rules:
- **Cricket/golf/football**: penalise rain in the previous 12 h (wet outfield); lightning gate.
- **Tennis**: wind > 20 km/h strong penalty; hard-court heat penalty at midday.
- **Hiking**: rain in the previous 24 h → slippery (monsoon ghats such as Lonavala); must finish 45 min before sunset; elevation gain; travel time weighs more (a long trip is acceptable).
- **Photography**: golden hour (astral) bonus; cloud cover 20–60 % at sunset bonus; visibility/haze (PM2.5 high → hazy); water/viewpoint bonus.
- **Outdoor event**: evaluate the **whole event span**, so max precip prob, gusts (tents) and heat for guests all matter; venue capacity proxy = area.
- **Picnic**: shade (green fraction), rain, heat.
- **Cycling**: gusts, route exposure to major roads, rush-hour traffic.
- **Running/walking/yoga**: AQ weight high; morning/evening preference; lit paths after dark.

**Profile modifiers** (tighten, never loosen): asthma (US AQI ≤ 100 → caution, > 100 → gate; avoid upwind roads; carry inhaler), senior ≥ 60 / "my parents" (apparent temp max −4 °C, avoid 11:00–16:00, short travel, flat), child/kids (UV ≥ 6 → shade/time shift, heat −3 °C, daylight only), pregnancy (apparent temp ≤ 32, AQ ≤ 100), cardiac (heat −4 °C, AQ ≤ 100, low exertion), heat-sensitive (−4 °C). Thresholds live in one table with a source comment (WHO/CPCB NAQI/US EPA AQI/heat-index bands) so we can defend them.

### 5.8 Grid scoring (`scoring/grid.py`)

- Cells: each candidate venue × each hour in the resolved windows (for a duration > 1 h, evaluate rolling windows: the gate applies if *any* hour fails, the score is the mean).
- Gate → cell status `avoid` with reasons. Otherwise score 0–100 = Σ wᵢ·sᵢ over thermal, rain, wind, UV (reduced by shade), air (sensitivity × profile), exposure (1 − index), setting (green/water/quiet), access (rush-hour-adjusted travel), light (daylight/golden hour).
- Output: best (venue, window); two alternatives at **different venues**; best alternative *time* at the top venue; a Saturday-vs-Sunday comparison when `compare`; avoid-windows merged into ranges ("avoid 11:00–16:00: apparent temp 38 °C [E7]"). If every cell is gated, say "not recommended in this period" and give the least-bad option with explicit risks.

### 5.9 Advice rules (`advice/rules.py`)

Each item = `{text_key, params, evidence_ids}`:
- **Carry**: water ≈ 0.5 L per 30 min (×1.5 if apparent ≥ 32 °C); electrolytes if > 90 min and ≥ 30 °C; sunscreen + cap if max UV ≥ 6; rain shell/umbrella if max precip prob ≥ 40 %; N95 if PM2.5 > 35 (sensitive) / > 60 (all); inhaler (asthma); headlamp if the window ends within 30 min of sunset; repellent if water ≤ 300 m in the evening; ID/emergency contact (seniors); snacks/hats (kids); spare battery/ND filter (photography); tarp/tent weights if gusts ≥ 30 (events).
- **Watch for**: heat-stress signs, lightning (any weather code 95–99 that day), gust spikes, slippery trails after rain, traffic near arterial roads, sunset time, AQ trend rising/falling within the window.
- **Avoid**: gated hour ranges with reasons; rush-hour travel if it would exceed the user's travel limit.

### 5.10 Fact sheet + composer (`evidence/ledger.py`, `slm/composer.py`)

- Ledger: every fetched or computed value becomes `Evidence(id="E7", var, value, unit, venue, hour, source, url, retrieved_at, kind=forecast|observation|osm|computed|modelled)`.
- Fact sheet (≈600 tokens) contains only what the answer needs: user ask summary, chosen venue/window, 2 alternatives, key per-hour values, advice items, limitations, each tagged `[E#]`.
- Composer prompt: "Use ONLY facts below. Cite [E#] after each fact. If a fact is missing, say it's unavailable. ≤220 words. Sections: …". Qwen3 with `enable_thinking=False`, temperature 0.3, max 260 tokens.

### 5.11 Grounding verifier (`evidence/grounding.py`)

- Extract numbers with units (°C, %, km/h, km, min, µg/m³, AQI, m), clock times, weekdays/dates, and Title-Case spans (place names).
- Each number must match a ledger value (tolerance ±0.5 or the rounding rule); times must lie within resolved windows or come from ledger sunrise/sunset; place names must fuzzy-match (≥0.85) candidate venues or the geocoded area; every `[E#]` must exist.
- Result `grounding: {status: passed|repaired|templated, violations: [...]}`. Repair = re-prompt once listing the violations; then fall back to the template renderer (`advice/template.py`), which builds the same sections deterministically.
- **This is our headline differentiator in the slides:** "every number in every answer is machine-checked against the data we fetched."

### 5.12 API (`api/ask.py`)

- `POST /api/v1/ask` → `AskResponse` (Section 4 of CONNECTIONS.md has the full contract). `?format=text` (or `Accept: text/plain`) returns just the markdown for curl-based judging.
- `GET /api/v1/ask/stream?q=…&session_id=…` → SSE events `stage`, `clarify`, `token`, `final`.
- `GET /api/v1/health` → **real** upstream probes (cached 60 s) + model loaded + version. `GET /api/v1/activities` stays.
- Keep `POST /api/v1/recommend` as a thin adapter over the new pipeline (frontend compat), delete `/explain` (hard-coded) or reimplement it over the ledger.
- Sessions: in-memory `{session_id: last_intent, last_ledger}` with a 30 min TTL for follow-ups ("what about Sunday?", "something closer?", "I'm taking my kids").
- Request timeout 60 s, per-IP rate limit (slowapi, 20/min), CORS restricted to the frontend origin + `*` for GET health.

### 5.13 New backend layout

```
backend/app/
  slm/            runtime.py (llama.cpp wrapper, warm-up, fallback chain) · prompts.py · intent_schema.py · composer.py
  agent/          understand.py · time_resolver.py · clarify.py · session.py · pipeline.py (replaces orchestrator.py)
  data_services/  http.py (shared clients, rate limiter, retries) · disk_cache.py · geocoding_service.py · overpass.py
                  weather_service.py · air_quality_service.py · waqi_service.py · routing_service.py · elevation_service.py
  geospatial/     projection.py · context.py · exposure.py · daylight.py
  scoring/        activity_profiles.py · profile_modifiers.py · grid.py · traffic.py · safety_constraints.py (refactored)
  advice/         rules.py · template.py
  evidence/       ledger.py · grounding.py
  api/            ask.py · health.py · activities.py · recommend.py (adapter)
```

New deps: `shapely>=2`, `astral`, `diskcache`, `llama-cpp-python`, `tzdata`, `rapidfuzz`, `slowapi`, `sse-starlette`. Dev: `respx` (mock httpx in tests).

---

## 6. Frontend plan (React 19 + Vite + Leaflet, keep the existing shell)

1. **Remove** `getFallbackRecommendation`; show a real error/retry state. Fix `VITE_API_URL` default to `:8000`.
2. **Chat panel** (left): conversation thread, streaming stage messages, clarification **quick-reply chips**, follow-ups using `session_id`, example prompts, a "Use my location" button (browser geolocation → `user_location`).
3. **Map** (centre): candidate venues coloured by score at the selected hour; recommended venue highlighted; layers you can toggle for major roads, industrial areas, water, green; a **wind arrow** for the selected hour; the search radius circle; a route line to the venue (OSRM geometry). Attribution "© OpenStreetMap contributors".
4. **Time scrubber + venue×hour heatmap** (bottom): drag through hours, so markers recolour and the wind arrow rotates. Gated cells are hatched, with the reason on hover. This *visually proves* "place and time together".
5. **Answer card** (right): best pick + window, the SLM's markdown with citation chips `[E7]` (click → evidence drawer highlights the raw value, source, URL, retrieved_at), Carry / Watch / Avoid lists, alternatives, a grounding badge ("✔ 14/14 facts verified" or "template used"), data freshness.
6. **Hourly chart** for the chosen venue: apparent temp, rain %, US AQI, exposure index (inline SVG; no heavy chart lib).
7. Mobile-friendly stacked layout; dark/light.
8. Build is served by the backend (`StaticFiles` at `/`) so judges get **one URL** for UI + API.

---

## 7. Evaluation & tests

- **Unit**: time resolver (60 cases incl. Hinglish, IST midnight edge, weekend on a Saturday), exposure math (wind flip changes ranking), profile tightening, grid gating, advice triggers, grounding verifier (catches injected fake numbers/places), intent schema validation.
- **Contract tests** with `respx` fixtures of real recorded API responses (`tests/fixtures/*.json`). No network in CI.
- **No-fabrication tests**: a grep-style test fails if `synthetic|baseline fallback|simulated` appears in `app/`; with Overpass mocked to fail, the answer must say "venue data unavailable" and contain no venue names.
- **Eval harness** `eval/run_eval.py` over `eval/queries.jsonl` (80 queries: 13 activities, 6 profiles, 15 messy/out-of-scope, 10 follow-ups). Metrics: intent slot accuracy, clarification correctness, groundedness pass rate (target 100 % after repair/template), % answers using SLM vs template (target ≥ 90 % SLM), p50/p95 latency, upstream failure counts. Run before/after fine-tune and put the table in the model card + slides.
- **Live smoke** (`scripts/smoke.sh`): 10 curls against the deployed URL; run every hour during judging via cron/GitHub Action.

---

## 8. Deployment

Single Docker image (multi-stage: build frontend → copy `dist` into the FastAPI image; download GGUF from HF at build or first boot). Needs about 3 GB RAM.

| Option | Plan | Notes |
|---|---|---|
| **A: Hugging Face Docker Space** (CPU basic 2 vCPU/16 GB) | Primary if your account can create a CPU-basic Docker Space; **check on Day 1** | Mid-2026 community reports say free CPU-basic/Docker Spaces are restricted for some new accounts. Spaces sleep after inactivity, so ping them |
| **B: Oracle Cloud Always Free** (Ampere 4 OCPU/24 GB) | Primary if A is unavailable; Caddy for HTTPS + a free DuckDNS/nip.io domain | Faster CPU inference; signup needs a card for verification (no charge) |
| **C: Hot standby**: laptop or Colab + Cloudflare **named** tunnel | Same image; stable hostname; documented in README as a mirror | Only if A/B are down |

- UptimeRobot (free) pings `/api/v1/health` every 5 min (keeps the Space awake + alerts us).
- Deploy v1 on **Day 2 night**, so hosting problems surface early (the FAQ explicitly warns about this).
- Secrets via env: `WAQI_TOKEN`, `ORS_API_KEY`, `NOMINATIM_EMAIL`.

---

## 9. Schedule (IST): 3.5 days, fine-tune is never on the critical path

| When | Track A: data & engine | Track B: model | Track C: UI, deploy, docs |
|---|---|---|---|
| **Tue 21 (today, rest of day)** | P0 blockers B1–B11 (≈3 h) · shared http/rate-limit/disk cache · AQ hourly fix · multi-coord weather · Nominatim fix · combined Overpass + mirror race | Create HF repos · bake-off of base models (1.5 h) · write `gen_intent_data.py` | Check HF Space / Oracle availability · fix API port · remove fake frontend fallback |
| **Wed 22** | TimeResolver · clarify/scope · features (shapely) · exposure · activity profiles · grid · advice rules · ledger + grounding | SLM runtime (llama.cpp) · intent + composer prompts with **base** model, wired end-to-end · start `gen_compose_data.py` (record raw API responses) | `/ask` + SSE · Docker multi-stage · **deploy v1 tonight** · UptimeRobot |
| **Thu 23** | Eval harness, fix top failures · test suite · perf (cache, timeouts) | **Colab LoRA fine-tune** (morning) · eval before/after · merge · GGUF · push 3 repos + model card · redeploy with FT | Chat UI, map layers, time scrubber + heatmap, evidence drawer, hourly chart |
| **Fri 24** | Hardening: failure-injection tests, rate limits, logs | Model card polish, final eval table | README, slides (4–6), demo script + backup screen recording · **code freeze 18:00** · submit form by **21:00** (3 h buffer) |

Gate checks: **end of Tue**, live data only, no fabrication, AQ real. **End of Wed**, a public URL answers all 10 example *styles* end-to-end with the base model. **End of Thu**, FT model live and eval table filled. If the fine-tune underperforms the base model on the eval, ship the base model with a config-only HF repo (allowed by the rules) and say so honestly.

---

## 10. Phase checklists (definition of done)

**P0: Truth fixes (Tue)**
- [ ] Delete all synthetic/baseline/fallback-fabrication code (B1, B2, B6, B9, B11); a factor that's unavailable becomes `available:false` + an answer note
- [ ] AQ: `/v1/air-quality`, hourly variables, multi-coord (B3)
- [ ] Geocoding: no Bengaluru default, `countrycodes=in`, contact UA, return candidates for ambiguity (B4, B5)
- [ ] `MODEL_NAME` → `Qwen/Qwen3-1.7B` everywhere incl. README (B8)
- [ ] Industrial analysis receives hourly wind (B7) (superseded by exposure.py in P2)
- [ ] Evidence method strings describe what we actually do (B10)
- [ ] tz-aware datetimes; IST "now"
- [ ] Test: no-fabrication test passes

**P1: Data layer**: shared clients + rate limiter + disk cache; Overpass combined query + mirror race + Geofabrik fallback (stretch); OSRM table; WAQI; OpenTopoData; `respx` fixtures recorded.

**P2: Engine**: time resolver, clarify, features, exposure, profiles (13 activities), grid, advice, ledger, grounding, template renderer. Tests for each.

**P3: SLM**: runtime + JSON-schema intent + composer + repair loop + fallback chain + `model.path_used` in the response. Sessions/follow-ups.

**P4: Fine-tune**: datasets A/B built + pushed; Colab notebook runs top-to-bottom; eval before/after; 3 HF repos; model card; `from_pretrained` verified in a clean Colab cell.

**P5: Frontend**: items 1–8 in Section 6.

**P6: Ship**: Docker, deploy, uptime monitor, smoke script, README, slides, disclosures, demo rehearsal (3 queries: asthma walk Powai, Sat vs Sun Lonavala hike, missing-location cycling → clarification).

---

## 11. Submission artefacts

**README.md** (rewrite): what it does, live URL, a curl example, architecture image, how to run (local/Docker), **base model = Qwen/Qwen3-1.7B + LoRA (link the 3 HF repos)**, data sources table (Section 4), © OpenStreetMap contributors, limitations, disclosures (AI coding assistants used: name them; open teacher model used for data generation: Qwen3-8B; no hosted LLM at runtime), licence.

**Slides (5)**
1. Problem → "the right answer depends on place × time × person". Show the expressway 8 AM vs evening example on our heatmap.
2. Architecture: the Section 2 diagram; SLM for language, engine for numbers, verifier for honesty.
3. Worked example: "I have mild asthma, morning walk near Powai in the next two days": parsed intent → map with roads/industry upwind → heatmap → answer with citations.
4. Model: Qwen3-1.7B, LoRA recipe, dataset sizes, eval before/after table, latency on CPU.
5. Limitations, stated honestly: CAMS ~40 km AQ grid, no live traffic (heuristic), OSM completeness varies across Indian cities, forecast uncertainty, Overpass availability, small-model phrasing quality.

**Judging Q&A prep**: Why not let the SLM rank? (Arithmetic and calendars are unreliable at 1.7B, and safety gates must be auditable.) How do you know it didn't hallucinate? (Verifier + grounding badge + evidence drawer, demoed live.) What if an API is down? (We show "unavailable", never invent; demo it by killing Overpass.) Did you hard-code examples? (Eval set differs; generator code is public.)

---

## 12. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Overpass mirrors down or slow during judging | **High** (seen 21 Oct) | Mirror race, 24 h cache of map features, Geofabrik-derived fallback store, honest "unavailable" path |
| Free hosting unavailable / sleeps | Medium | Check Day 1; Oracle Always Free; Cloudflare-tunnel standby; UptimeRobot |
| CPU inference too slow | Medium | Q4_K_M, compact fact sheet, 260-token cap, SSE streaming, 4-core host |
| Fine-tune worse than base | Medium | Eval gate; ship base + prompts; FT never blocks the deadline |
| Nominatim rate limit / ambiguity | Medium | 1 req/s limiter, 30-day cache, ask the user when ambiguous |
| Small model ignores citations / invents numbers | Medium | Constrained sections, verifier + repair + template fallback |
| Weekend/date logic bugs | Medium | Resolver unit tests incl. timezone edges |
| Scope creep (commute corridors, OpenAQ, CPCB) | High | Marked *stretch*; only after P0–P6 are green |
