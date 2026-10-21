# CONNECTIONS.md — how everything in TerraPulse AI connects

Wiring reference for the system: external APIs, internal modules, model artifacts, frontend↔backend contract, env vars, ports and failure behaviour. Plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Rules: [AGENTS.md](AGENTS.md).

**State (21 Oct 2026, evening): everything below is implemented** unless marked *(planned)*. Endpoints marked "verified" were called live from the dev machine.

---

## 1. Topology

```
 Browser (React/Vite/Leaflet)                     Judges (curl / Postman)
        │  SSE + JSON                                     │ JSON or text
        ▼                                                 ▼
 ┌──────────────────────── FastAPI  (port 8000, one container) ────────────────────────┐
 │  /  → frontend dist (StaticFiles)                                                │
 │  /api/v1/ask, /ask/stream   /recommend   /health   /activities    │
 │                                                                                    │
 │  agent/pipeline.py ──► slm/runtime.py ──► llama.cpp ──► GGUF (from HF Hub)          │
 │        │                                                                           │
 │        ├─► data_services/* ──► http.py (shared clients, rate limiter, retries)     │
 │        │                         └─► cache.py (in-memory TTL, geohash keys)         │
 │        ├─► geospatial/* (shapely)   scoring/*   advice/*   evidence/*              │
 └────────┼───────────────────────────────────────────────────────────────────────────┘
          ▼  outbound HTTPS only (no inbound dependencies)
   Nominatim · Overpass (mirrors) · Open-Meteo forecast · Open-Meteo air-quality ·
   OSRM · WAQI · OpenTopoData · (ORS, OpenAQ, data.gov.in = optional)
```

---

## 2. External services: exact calls

| Service | Module | Request (verified form) | Parse | Cache key / TTL | On failure |
|---|---|---|---|---|---|
| **Nominatim** | `data_services/geocoding_service.py` | `GET https://nominatim.openstreetmap.org/search?q=Powai&countrycodes=in&format=jsonv2&limit=5&addressdetails=1` with header `User-Agent: TerraPulse-hackathon/0.1 ($NOMINATIM_EMAIL)` (verified 21 Oct: Powai → 19.1187, 72.9073) | lat, lon, boundingbox, display_name, address.city/state, type | `geo:{q_lower}` / 30 d | Return `clarification_needed` ("couldn't find that place"); **never a default city** |
| **Overpass** | `data_services/osm_service.py` | `POST {mirror}/api/interpreter` form `data=[out:json][timeout:25][maxsize:67108864]; ( …candidate + context statements… ); out geom;` | elements → shapely Points/LineStrings/Polygons + tags | `osm:{activity_group}:{lat3}:{lon3}:{radius_km}` / 24 h | Rolling mirror race (§2.1) + 5-min circuit breaker → **Nominatim category search** (`q=<phrase>&viewbox&bounded=1`, phrases per activity in activities.yaml) → else area-only answer with no venue names |
| **Open-Meteo forecast** | `data_services/weather_service.py` | `GET https://api.open-meteo.com/v1/forecast?latitude=19.11,19.13&longitude=72.90,72.88&hourly=temperature_2m,apparent_temperature,relative_humidity_2m,precipitation_probability,precipitation,weather_code,cloud_cover,visibility,wind_speed_10m,wind_direction_10m,wind_gusts_10m,uv_index,is_day&daily=sunrise,sunset,uv_index_max&past_days=1&forecast_days=6&timezone=Asia/Kolkata` (multi-coord verified 21 Oct; returns a JSON **list**, one object per coord) | per venue: list of hourly dicts; daily sunrise/sunset | `wx:{sorted coords hash}:{start}:{days}` / 30 min | Factor `weather` unavailable → **no recommendation** (weather is mandatory); explain and suggest retry |
| **Open-Meteo air quality** | `data_services/air_quality_service.py` | `GET https://air-quality-api.open-meteo.com/v1/air-quality?latitude=…&longitude=…&hourly=pm2_5,pm10,nitrogen_dioxide,ozone,us_aqi,us_aqi_pm2_5,us_aqi_pm10,us_aqi_ozone,us_aqi_nitrogen_dioxide&start_date=…&end_date=…&timezone=Asia/Kolkata` (path has a **hyphen**; verified 200; old `/air_quality` = 404) | hourly arrays per coord | `aq:{coords hash}:{start}` / 60 min | Factor `air_quality` unavailable, weight redistributed, the answer states it; asthma profile → recommend checking a live AQI before going |
| **WAQI** | `data_services/waqi_service.py` | `GET https://api.waqi.info/feed/geo:{lat};{lon}/?token=$WAQI_TOKEN` | station name, aqi, iaqi.pm25, time.iso, station distance | `waqi:{lat2}:{lon2}` / 30 min | Skip silently (supplementary); evidence shows "no ground station" |
| **OSRM** | `data_services/routing_service.py` | `GET https://router.project-osrm.org/table/v1/driving/{oLon},{oLat};{v1Lon},{v1Lat};…?sources=0&annotations=duration,distance` (verified 21 Oct) and `/route/v1/driving/…?overview=simplified&geometries=geojson` for the chosen venue only | durations[0][i] s, distances[0][i] m | `osrm:{origin3}:{dest-set hash}` / 6 h | ORS matrix if `ORS_API_KEY`; else straight-line distance shown **as distance only**, travel time `unknown` (no invented minutes) |
| **OpenTopoData** *(planned)* | `data_services/elevation_service.py` | `GET https://api.opentopodata.org/v1/srtm30m?locations=lat,lon|lat,lon…` (≤100) | elevation m | `elev:{lat4}:{lon4}` / 30 d | Omit elevation facts |
| Golden hour | `scoring/grid.py` | computed from the **live** Open-Meteo sunrise/sunset (window in scoring.yaml) | — | — | no light bonus if sunrise/sunset missing |
| ORS (optional) | routing fallback | `POST https://api.openrouteservice.org/v2/matrix/driving-car` header `Authorization: $ORS_API_KEY` | durations | same as OSRM | skip |

### 2.1 Overpass mirror race (`overpass.py`)
Mirrors in config `OVERPASS_MIRRORS` (comma list). Fire the first 2 concurrently, take the first valid JSON with `elements`, cancel the rest; if both fail within 25 s, try the remaining ones sequentially. Probe on 21 Oct from the dev machine: `overpass-api.de` connection failed, `overpass.private.coffee` timed out, `overpass.kumi.systems` (POST) timed out, **`maps.mail.ru/osm/tools/overpass` returned 200 in 22 s**. Re-probe from the deploy host.
Order (config `OVERPASS_MIRRORS`): overpass-api.de, maps.mail.ru, overpass.kumi.systems, overpass.private.coffee. Two in flight; a failure starts the next; first valid answer wins (`geo.yaml: overpass_rolling_race`). Per-request timeout `OVERPASS_TIMEOUT_S=25`; after all fail, Overpass is skipped for `OVERPASS_BREAKER_S=300`.

### 2.2 Rate limits enforced in `http.py`
`HOST_POLICIES` in `main.py`: Nominatim ≥1.05 s spacing, 1 concurrent · OSRM 1/s · Overpass ≤2 concurrent per mirror · Open-Meteo 5/s. Retries: 2, full-jitter exponential backoff (base 0.6 s, cap 6 s), honours `Retry-After`; 4xx other than 429 are not retried.

---

## 3. Internal module wiring (request lifecycle)

```
api/ask.py
 └─ agent/pipeline.py: run(question, session, user_location) -> AskResponse
     1  slm/runtime.intent(question, prev_intent)            -> IntentJSON          (fallback: agent/intent_parser.py regex)
     2  agent/time_resolver.resolve(intent.time_text, tz)     -> [TimeWindow], compare
     3  agent/clarify.check(intent, windows, user_location)   -> Clarification | None   ← early return
     4  data_services/geocoding.geocode(location_text)        -> Place(lat, lon, bbox, candidates)
     5  asyncio.gather(
          overpass.fetch_area(activity, place, radius),       -> OSMBundle(candidates, roads, industry, water, green, power)
          weather.forecast(points=[place]),                   -> early sunrise/sunset for resolver refinement
        )
     6  scoring/activity_profiles.select_candidates(OSMBundle) -> ≤8 venues
     7  asyncio.gather(weather.forecast(venues), air_quality.hourly(venues),
                       routing.table(origin, venues), waqi.nearest(place), elevation(venues) if hiking)
     8  geospatial/context.features(venue, OSMBundle)          -> StaticFeatures
        geospatial/exposure.hourly(venue, OSMBundle, wind[h])  -> exposure_index[h]
        geospatial/daylight.for_day(place, day)                -> sunrise/sunset/golden hours
     9  scoring/grid.build(venues × hours, profile⊕modifiers)  -> Grid, best, alts, avoid
    10  advice/rules.build(best, grid, profile)                -> Carry/Watch/Avoid items
    11  evidence/ledger.build(all above)                        -> Ledger (E1…En), FactSheet
    12  slm/composer.write(FactSheet)                           -> markdown
    13  evidence/grounding.verify(markdown, Ledger)            -> passed | repair (→12 once) | template (advice/template.py)
    14  agent/session.save(session_id, intent, ledger)
```

Shared singletons created once in `main.py` lifespan and stored on `app.state`: `HttpGateway`, `InMemoryCache`, `SLMRuntime`, `Pipeline` (which owns all services and the `SessionStore`).

---

## 4. Frontend ↔ backend contract

Base URL: `VITE_API_URL`, default relative `/api/v1`. In dev, Vite proxies `/api` → `http://localhost:8000`; in production FastAPI serves `frontend/dist` at `/`.

### 4.1 `POST /api/v1/ask`
Request:
```json
{ "question": "I have mild asthma, where's good for a morning walk near Powai in the next two days?",
  "session_id": "optional-uuid",
  "user_location": {"lat": 19.12, "lon": 72.90},          // optional (browser geolocation)
  "overrides": {"radius_km": 10, "max_travel_min": 30}     // optional UI controls
}
```
Response (`AskResponse`):
```json
{
  "status": "ok | clarification_needed | out_of_scope | not_recommended | partial | error",
  "session_id": "uuid",
  "answer_markdown": "**Best pick:** … [E3] …",
  "clarification": {"question": "Which area should I look around?", "options": ["Use my location", "Powai, Mumbai", "Powai, Bengaluru"]},
  "intent": { "...IntentJSON..." , "resolved_windows": [{"day":"2026-10-22","start":"05:30","end":"10:00","label":"morning"}] },
  "recommendation": {
    "venue": {"id":"osm-way-123","name":"…","lat":19.12,"lon":72.91,"type":"park","tags":{}},
    "window": {"start":"2026-10-22T06:00+05:30","end":"2026-10-22T07:00+05:30"},
    "score": 84.2, "safety": "ok | caution",
    "travel": {"minutes": 9, "minutes_rush_adjusted": 14, "km": 3.1, "source": "OSRM"},
    "factors": [{"name":"air","score":0.71,"weight":0.3,"available":true,"evidence":["E5","E6"]}]
  },
  "alternatives": [ { "...same shape...": "" } ],
  "compare": [{"label":"Saturday","best_score":78},{"label":"Sunday","best_score":64}],
  "avoid": [{"start":"…","end":"…","reasons":["US AQI 152 [E9]"]}],
  "advice": {"carry":[{"text":"…","evidence":["E4"]}], "watch_for":[], "avoid":[]},
  "grid": {"venues":["v1","v2"], "hours":["…"], "scores":[[84,80,null]], "gates":[[null,null,"rain 70%"]]},
  "geojson": {"type":"FeatureCollection","features":[ "venues, chosen route, major roads, industry, water, radius" ]},
  "evidence": [{"id":"E5","var":"pm2_5","value":38.1,"unit":"µg/m³","venue":"v1","time":"…","kind":"forecast",
                "source":"Open-Meteo Air Quality (CAMS)","url":"https://air-quality-api.open-meteo.com/v1/air-quality?…","retrieved_at":"…"}],
  "sources": ["© OpenStreetMap contributors", "Open-Meteo", "OSRM", "WAQI"],
  "limitations": ["AQ forecast grid ~40 km: venue differences come from modelled exposure", "Travel times exclude live traffic"],
  "grounding": {"status":"passed | repaired | templated", "facts_checked": 14, "violations": []},
  "model": {"repo":"aaronr43210/terrapulse-qwen3-1.7b-GGUF","file":"…Q4_K_M.gguf","path_used":"finetuned | base | template"},
  "timings_ms": {"intent":2100,"fetch":6400,"score":120,"compose":9800,"total":18900}
}
```
`?format=text` or `Accept: text/plain` returns `answer_markdown` (+ sources footer) only.

### 4.2 `GET /api/v1/ask/stream?question=…&session_id=…`
SSE events: `stage` `{name, message}` → optional `clarify` → `token` `{t}` (composer tokens) → `final` (full AskResponse). The frontend `services/api.ts` uses `EventSource`; the `RecommendationResponse` type maps from `AskResponse` (update `types/index.ts`).

### 4.3 Others
`POST /api/v1/recommend` (alias: `{query, latitude, longitude, radius_km, duration_minutes, profiles}`) · `GET /api/v1/explain?session_id=` (fact sheet + evidence + grounding of the last answer) · `GET /api/v1/venues?activity=&place=` · `GET /api/v1/ask?q=&format=text` · SSE also accepts `profiles=asthma,senior` and `lat/lon`.
`GET /api/v1/health` → `{status, model_loaded, model_path_used, upstream:{nominatim:"ok|degraded|down", …}, version}` (real probes, cached 60 s). `GET /api/v1/activities` → profiles. `POST /api/v1/recommend` → adapter that calls the pipeline and maps to the old shape (until the UI migrates).

---

## 5. Model artifacts ↔ runtime

```
Qwen/Qwen3-1.7B (HF, Apache-2.0)
   │  Colab (VS Code Colab extension, T4) · training/02_finetune_lora_colab.ipynb (data from 01_build_data_colab.ipynb) · Unsloth LoRA
   ├─► aaronr43210/terrapulse-qwen3-1.7b-lora     adapter_model.safetensors + adapter_config.json(base_model_name_or_path)
   ├─► aaronr43210/terrapulse-qwen3-1.7b          merged safetensors → from_pretrained() works   (judges)
   └─► aaronr43210/terrapulse-qwen3-1.7b-GGUF     *.Q4_K_M.gguf, *.Q8_0.gguf                       (server)
                                                    │ huggingface_hub.hf_hub_download at build/boot
                                                    ▼
                                    backend/app/slm/runtime.py (llama-cpp-python)
```
Config: `SLM_REPO`, `SLM_FILE`, `SLM_BASE_REPO`/`SLM_BASE_FILE` (fallback GGUF of the base model: `unsloth/Qwen3-1.7B-GGUF` / `Qwen3-1.7B-Q4_K_M.gguf`, verified on the Hub), `SLM_THREADS`, `SLM_CTX=4096`.
Training data repos (optional): `aaronr43210/terrapulse-intent-data`, `aaronr43210/terrapulse-compose-data`.

---

## 6. Environment variables (`backend/.env`, mirror in `.env.example`)

| Var | Default | Purpose |
|---|---|---|
| `APP_TZ` | `Asia/Kolkata` | fallback timezone for "now" |
| `NOMINATIM_URL` | `https://nominatim.openstreetmap.org` | |
| `CONTACT_EMAIL` | aaronr43210@gmail.com | contact in User-Agent (Nominatim policy) |
| `OVERPASS_MIRRORS` | see §2.1 | comma-separated |
| `WEATHER_API_URL` | `https://api.open-meteo.com/v1` | |
| `AIR_QUALITY_API_URL` | `https://air-quality-api.open-meteo.com/v1` | path `/air-quality` appended in code |
| `OSRM_URL` | `https://router.project-osrm.org` | |
| `ORS_API_KEY` | empty | optional routing fallback |
| `WAQI_TOKEN` | empty | optional observed AQI |
| `ELEVATION_API_URL` | `https://api.opentopodata.org` | |
| `OVERPASS_TIMEOUT_S` / `OVERPASS_BREAKER_S` | 25 / 300 | per-mirror timeout / skip window after total failure |
| `SLM_ENABLED` / `SLM_BACKEND` | `true` / `llama_cpp` | `transformers` backend also supported; `SLM_LOCAL_PATH` overrides downloads |
| `SLM_REPO` / `SLM_FILE` | `aaronr43210/terrapulse-qwen3-1.7b-GGUF` / `…Q4_K_M.gguf` | |
| `SLM_BASE_REPO` / `SLM_BASE_FILE` | base GGUF | fallback |
| `SLM_THREADS` | cpu count | |
| `CORS_ORIGINS` | `http://localhost:5173` | |
| `VITE_API_URL` (frontend) | `/api/v1` (prod) / `http://localhost:8000/api/v1` (dev) | |

Remove: `MODEL_NAME=microsoft/phi-3-mini-4k-instruct` (3.8B, over the limit), `DEVICE`, `ENABLE_LOCAL_SLM`.

---

## 7. Ports & processes

| Process | Dev | Prod |
|---|---|---|
| FastAPI (uvicorn) | `:8000` | `:7860` in the container (`PORT`), HF Space `app_port: 7860`; compose maps host 8000 → 7860 |
| Vite dev server | `:5173` | n/a (dist served by FastAPI) |
| docker-compose | single service `terrapulse` | UI + API + model in one image |

---

## 8. Failure-behaviour matrix (no fabrication, ever)

| Upstream down | User sees | `status` |
|---|---|---|
| Nominatim | "I couldn't locate *X*; can you name a nearby area or share your location?" | `clarification_needed` |
| Overpass (all mirrors) | Venues from Nominatim category search; roads/industry/green "unavailable" (never 0) | `partial` |
| Overpass and Nominatim venues | Timing advice for the *area* only, no venue names | `partial` |
| Open-Meteo forecast | "Live forecast unavailable; I can't recommend a time safely. Please retry." | `error` |
| Open-Meteo AQ | Recommendation without the AQ factor + explicit note (+ asthma caution) | `partial` |
| OSRM (and ORS) | Distances only, "travel time unavailable" | `partial` |
| WAQI / OpenTopoData | Omitted facts, nothing else changes | `ok` |
| SLM fails / too slow | Template-rendered answer, `model.path_used = template` | `ok` |

---

## 9. Attribution strings (show in UI footer + README + API `sources`)
"Map data © OpenStreetMap contributors (ODbL)" · "Weather & air quality: Open-Meteo.com (CC BY 4.0), CAMS" · "Routing: OSRM / Project OSRM" · "AQI stations: World Air Quality Index Project" · "Elevation: OpenTopoData (SRTM)".
