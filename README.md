# TerraPulse AI

**Where and when to head outdoors, for you, grounded in live open data.**
GeoAI Hackathon @ CSRE, IIT Bombay · small-language-model assistant for outdoor activities.

Ask in plain words (English or Hinglish): *"I have mild asthma, where's good for a morning walk near ‹area› in the next
two days?"*. TerraPulse finds real nearby places on OpenStreetMap, scores every place across every hour of your time
window using live weather, air quality, wind-aware road/industry exposure, green cover, water and travel time, then
explains the best place × time, what to carry, what to watch for and when to avoid going. Every number in the answer
is cited and machine-checked against the data it fetched.

| | |
|---|---|
| **Live API** | `‹your deployment URL›/api/v1/ask` (UI at `/`, docs at `/docs`) |
| **Model** | [`aaronr43210/terrapulse-qwen3-1.7b`](https://huggingface.co/aaronr43210/terrapulse-qwen3-1.7b) (merged, `from_pretrained`) · [`-lora`](https://huggingface.co/aaronr43210/terrapulse-qwen3-1.7b-lora) (adapter) · [`-GGUF`](https://huggingface.co/aaronr43210/terrapulse-qwen3-1.7b-GGUF) (served by the API) |
| **Base model** | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) |
| **Code** | https://github.com/aaron43210/GeoAI_Hackathon |
| **Licence** | Apache-2.0 |

## How it works

```
question ─► SLM: intent JSON (grammar-constrained) ─► deterministic time resolver (IST, Hinglish, weekends…)
        ─► clarify / scope gate (asks instead of guessing)
        ─► Nominatim geocode ─► OSM venues (Overpass, fallback: Nominatim category search)
        ─► live data in parallel: Open-Meteo hourly weather + sunrise/sunset · Open-Meteo/CAMS hourly air quality
                                   · OSRM travel matrix · OSM roads/industry/water/green/power around venues
        ─► features (shapely) + per-hour wind-aware exposure ─► venue × hour grid with 3-valued safety rules
        ─► advice rules (carry / watch for / avoid) ─► evidence ledger + fact sheet [E1…En]
        ─► SLM writes the answer citing [E#] ─► grounding verifier ─► (repair once ─► template)
```

* **SLM does language, code does arithmetic.** A 1.7B model reads messy questions and writes the advice. Scoring,
  safety gates and dates are deterministic and auditable. The model never needs to remember weather or places.
* **Place × time.** Each venue is scored for every hour. Exposure uses the forecast wind for that hour: a ground
  downwind of an expressway at 8 AM can be fine in the evening once the wind turns. The answer reports that contrast.
* **Nothing is invented.** No synthetic venues, simulated weather, baseline AQI or default city. If a source fails,
  the answer says so (`partial`), and without a forecast it refuses to recommend.
* **Grounding verifier** ([backend/app/evidence/grounding.py](backend/app/evidence/grounding.py)): numbers and units,
  clock times, place names, citation ids, citation-to-value adjacency, "not recommended" framing, and severity wording
  (e.g. calling AQI 175 "acceptable" fails).
* **All knowledge is data, not code.** Activities, OSM tags, thresholds, profiles, safety rules, advice, time words and
  scoring curves live in [backend/app/knowledge/*.yaml](backend/app/knowledge/) with their sources (US EPA AQI guidance,
  NOAA heat-index bands, WHO UV index). A test fails if example place names or fabrication code paths appear in runtime code.

### Activities (13)
Running · walking · cycling (leisure/commute) · golf · cricket · football/five-a-side · tennis · hiking · picnic ·
outdoor event (wedding, party, fair) · photography (golden hour) · yoga · birdwatching.
Profiles that only ever **tighten** limits: asthma, senior (60+ / "my parents"), children, pregnancy, heart condition,
heat-sensitive, limited mobility.

## API

```bash
# plain text answer
curl "‹URL›/api/v1/ask?q=golf+this+sunday+within+30+km+of+Gurugram&format=text"

# full JSON: answer, recommendation, alternatives, avoid windows, advice, heatmap, timeline, evidence, sources
curl -X POST ‹URL›/api/v1/ask -H 'content-type: application/json' \
     -d '{"question":"evening jog near Indiranagar Bengaluru tomorrow?"}'

# follow-up in the same session
curl -X POST ‹URL›/api/v1/ask -d '{"question":"what about Sunday?","session_id":"<from previous>"}' -H 'content-type: application/json'
```

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/ask` · `GET /api/v1/ask?q=` | main endpoint (`?format=text` for markdown only) |
| `GET /api/v1/ask/stream?q=` | server-sent events: progress stages, draft tokens, final answer |
| `POST /api/v1/recommend` | alias accepting `{query, latitude, longitude, radius_km, duration_minutes, profiles}` |
| `GET /api/v1/explain?session_id=` | fact sheet, evidence ledger and grounding report behind the last answer |
| `GET /api/v1/venues?activity=&place=` | live OSM venue discovery |
| `GET /api/v1/activities` | activities, profiles, limits, weights (from the knowledge base) |
| `GET /api/v1/health` | model status + live upstream probes |

Statuses: `ok`, `partial` (some source missing, said in the answer), `not_recommended` (every hour hits a safety
limit; the least-bad option and risks are given), `clarification_needed`, `out_of_scope`, `not_available` (beyond
forecast range / time passed), `error`.

## Run it

```bash
# backend (Python 3.11+)
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # core deps + llama-cpp-python
cp .env.example .env                       # set CONTACT_EMAIL
uvicorn app.main:app --port 8000           # downloads the GGUF on first start (~1.1 GB)

# frontend (dev; proxies /api to :8000)
cd frontend && npm install && npm run dev  # http://localhost:5173
# or build once and let FastAPI serve it at http://localhost:8000
npm run build

# tests (offline, upstream APIs mocked)
cd backend && pip install -r requirements-dev.txt && pytest -q

# live evaluation against a running server
python -m eval.run_eval --base http://localhost:8000

# single container (UI + API + model)
docker compose up --build                  # http://localhost:8000
```

## Model & fine-tuning (Colab only)
**Result of the fine-tune** (questions never seen in training, Q4_K_M on CPU):

| | base Qwen3-1.7B | TerraPulse fine-tune |
|---|---|---|
| activity / people-profile extraction (24 gold questions) | 0.917 / 0.958 | **1.0 / 1.0** |
| in/out-of-scope detection | 0.917 | **0.958** |
| answer drafts passing the grounding verifier first time (live) | 4 / 13 | **12 / 14** |
| correct status on live questions | 18 / 20 | **19 / 20**, and 23 / 24 on a second set |

* Model choice from a measured bake-off on CPU (Q4_K_M, 24 hand-written gold questions, not the hackathon examples):
  Qwen3-1.7B: activity 91.7%, location 100%, profiles 95.8%, 2.2 s/intent, 0 invented numbers;
  Qwen3.5-2B: 87.5% / 95.8% / 95.8%, 4.9 s. Excluded: Phi-3/4-mini (3.8 B), Ministral-3-3B (3.85 B),
  Granite-4.0-micro (3.4 B), Llama-3.2-3B (3.21 B, non-OSI licence), Qwen2.5-3B (non-commercial licence),
  SmolLM3 (3.08 B), Gemma-3n-E2B (5.4 B). Script: [backend/eval/bakeoff.py](backend/eval/bakeoff.py).
* Training runs **only in Google Colab** (VS Code Colab extension, free T4):
  1. [training/01_build_data_colab.ipynb](training/01_build_data_colab.ipynb): ~3,000 generated intent examples with exact
     labels + ~160 real fact sheets from the live pipeline; answer drafts from the open **Qwen3-4B** (fp16, batched), kept only if they
     pass the grounding verifier.
  2. [training/02_finetune_lora_colab.ipynb](training/02_finetune_lora_colab.ipynb): Unsloth LoRA (r=16, α=32, 2 epochs,
     completion-only loss), before/after eval, push adapter + merged model + GGUF + model card.
* `training/update_model_cards.py` publishes the model cards; the runtime loads the fine-tuned GGUF and falls back to the
  base GGUF only if it can't be downloaded.

## Data sources (all free and open, fetched live at query time)
| Source | Used for |
|---|---|
| OpenStreetMap via **Overpass API** (mirrors: overpass-api.de, maps.mail.ru, kumi.systems, private.coffee) | venues, roads, industry, water, green cover, power lines |
| OpenStreetMap **Nominatim** | geocoding; category search as venue fallback |
| **Open-Meteo** Forecast API | hourly temperature, feels-like, humidity, rain probability/amount, weather code, cloud, visibility, wind/gusts/direction, UV, day/night; daily sunrise/sunset |
| **Open-Meteo Air Quality** API (Copernicus CAMS) | hourly PM2.5, PM10, NO₂, O₃, US AQI + sub-indices |
| **OSRM** (Project OSRM demo server) | road travel time/distance matrix |
| **WAQI** (optional, free token) | nearest ground-station AQI |

Map data © OpenStreetMap contributors (ODbL). Weather data by Open-Meteo.com (CC BY 4.0).

## Honest limitations
* CAMS air quality is a ~40 km grid: venues in one city share the AQI forecast; venue differences come from the
  **modelled** wind × road/industry exposure index, labelled as modelled. Ozone-driven AQI is flagged as such.
* OSRM has no live traffic; a disclosed rush-hour multiplier (geo.yaml) is applied.
* OSM completeness varies; public Overpass servers can be slow, in which case Nominatim is used and map context
  (roads/industry/green) may be unavailable, which the answer states.
* Health limits follow published guidance but are not medical advice.
* Latency on a 4-core CPU: ~10–20 s typical, dominated by Overpass. Answers stream progressively.

## Disclosures
* Runtime model: Qwen3-1.7B (+ our LoRA), run locally with llama.cpp. No hosted LLM in the running assistant.
* Training-data teacher: Qwen3-4B (open weights, Apache-2.0), run in Colab.
* AI coding assistants were used during development (disclosed per the hackathon rules); no AI service runs in the product.

## Repo map
`backend/app/` FastAPI app (`agent/` pipeline, `data_services/` live APIs, `geospatial/`, `scoring/`, `advice/`,
`evidence/`, `slm/`, `knowledge/` YAML) · `backend/tests/` · `backend/eval/` · `frontend/` React + Leaflet ·
`training/` Colab notebooks, data generators, model-card tools · `deploy/` Space config ·
`docs/` [architecture & API contract](docs/ARCHITECTURE.md), [problem statement](docs/problem-statement.txt) ·
[LICENSE](LICENSE) (Apache-2.0).
