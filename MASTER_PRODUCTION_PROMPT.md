# MASTER PRODUCTION IMPLEMENTATION PROMPT
## SYSTEM: TerraPulse AI — Real-Time Evidence-Driven GeoAI Outdoor Activity Recommendation Engine
### Target Event: GeoAI Hackathon @ CSRE, IIT Bombay

---

```markdown
You are an expert Principal Geospatial AI Systems Engineer and Distributed Backend Architect.
Your task is to build a complete, production-grade, end-to-end outdoor activity recommendation system
named "TerraPulse AI" that adheres 100% to the rules of the GeoAI Hackathon at CSRE, IIT Bombay.

CRITICAL CONSTRAINTS & NON-NEGOTIABLE RULES:
1. ZERO HARDCODED ANSWERS: Do NOT hardcode mock locations, mock venues, mock weather, mock AQI,
   or pre-canned responses for the example queries. All data MUST be fetched live at runtime.
2. ZERO DATASET DEPENDENCY: No preloaded shapefiles, GeoJSON dumps, or static databases.
   Everything must be fetched dynamically at query time using free, open public APIs.
3. SMALL LANGUAGE MODEL ONLY (<= 3B Parameters): The NLP/reasoning component MUST run an open-weights
   model of <= 3 billion parameters (e.g., Qwen/Qwen2.5-3B-Instruct, meta-llama/Llama-3.2-3B-Instruct,
   or microsoft/Phi-3-mini-4k-instruct). NO hosted proprietary LLMs (No OpenAI GPT-4, Claude, Gemini).
4. GROUNDED EVIDENCE PROVENANCE: Every claim (temperature, wind, air quality, road exposure, green cover)
   must cite the exact live API endpoint, raw value, unit, timestamp, and mathematical formula used.
   If data is unavailable or rate-limited, explicitly report it as missing rather than hallucinating.
5. RESILIENCE & PRODUCTION QUALITY: Proper async I/O (httpx, asyncio), rate limiting with exponential
   backoff, in-memory TTL caching with cache keys based on spatial geohash/coordinates, Pydantic v2
   type validation, Docker containerization, comprehensive pytest suite, and a modern React + Leaflet UI.
```

---

## I. SYSTEM ARCHITECTURE & DATA FLOW

Implement the architecture according to this deterministic multi-stage pipeline:

```
[User Natural Language Query / Free-Form Input]
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. INTENT EXTRACTION & PARAMETER PARSING (Local 3B SLM)     │
│ • Model: Qwen/Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct  │
│ • Enforces JSON schema: Activity, Location, Time Window,    │
│   Duration, Health Profile (Asthma, Senior, Pregnancy, etc.)│
│   Spatial Preferences (Green cover, Road avoid, Industrial) │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. DYNAMIC GEOCODING & BOUNDING BOX CALCULATION             │
│ • Live OpenStreetMap Nominatim API (with User-Agent header)  │
│ • Computes (lat, lon) and radial bounding box               │
└──────────────────────────────┬──────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│ 3. LIVE VENUE DISCOVERY      │ │ 4. LIVE ATMOSPHERIC METRICS  │
│ • OpenStreetMap Overpass API │ │ • Open-Meteo Weather API     │
│ • Dynamically builds QL query│ │   (24h hourly forecast: temp,│
│   for activity-specific tags │ │    apparent temp, wind, rain,│
│   (parks, tracks, cycleways, │ │    gusts, UV, cloud cover)   │
│   pitches, viewpoints, etc.) │ │ • Open-Meteo Air Quality API │
│ • Filters real physical nodes│ │   (hourly PM2.5, PM10, NO2,  │
│   & ways with coordinates    │ │    O3, European AQI)         │
└──────────────┬───────────────┘ └──────────────┬───────────────┘
               │                                │
               └───────────────┬────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. ADVANCED GEOSPATIAL ANALYSIS (Live Vector Overpass API)  │
│ • Road Analysis: Distance to nearest highway & road density │
│ • Green Canopy: OSM leisure=park, natural=wood intersection │
│ • Industrial Buffer: Distance to landuse=industrial         │
│ • Wind Vector Advection: Project wind direction (0-360°)    │
│   relative to industrial bearing (calculates downwind plume)│
│ • Routing & Transit: OSRM engine routing & travel duration  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. DETERMINISTIC SAFETY FILTERING & MULTI-FACTOR RANKING    │
│ • Hard Safety Constraints: Strict physical bounds across all│
│   10 activities + multipliers for user health conditions.   │
│ • 24-Hour Diurnal Window Optimizer: Scans contiguous hourly │
│   blocks to find peak comfort and lowest precipitation.     │
│ • Multi-Factor Scoring Engine: Transparent weighted formula │
│   (Weather 35%, AQI 25%, Geospatial 20%, Travel 15%, Pref 5%)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. EVIDENCE VALIDATION & GROUNDED EXPLANATION GENERATION    │
│ • Compiles immutable EvidenceRecord objects for every claim │
│ • 3B SLM Explainer: Synthesizes plain-language guidance,    │
│   "what to carry", "what to watch for", and "when to avoid" │
│   referencing ONLY the verified evidence IDs.               │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. API GATEWAY & REACT + LEAFLET INTERFACE                  │
│ • FastAPI REST Endpoints (/api/v1/recommend, /activities...) │
│ • Interactive Map with live tiles, pins, 250m road buffer,   │
│   300m green canopy circle, 24-hour timeline, evidence audit│
└─────────────────────────────────────────────────────────────┘
```

---

## II. COMPONENT-BY-COMPONENT IMPLEMENTATION SPECIFICATION

### 1. Intent Extraction & SLM Engine (`app/agent/intent_parser.py`)
- **Model Requirement**: Use `transformers` pipeline with an open-weights <= 3B model (e.g. `Qwen/Qwen2.5-3B-Instruct` or `meta-llama/Llama-3.2-3B-Instruct` in 4-bit/8-bit or float16/float32 depending on device).
- **Structured Output**: Prompt the model with strict JSON formatting constraints:
  - `activity`: One of `[running, cycling, golf, cricket, football, tennis, hiking, picnic, outdoor_event, photography]`
  - `location_query`: Free-form location name extracted from the text (e.g. "Koregaon Park", "Powai", "Whitefield", "Lonavala")
  - `date_target`: Resolved target date (`today`, `tomorrow`, `weekend`, or ISO date)
  - `time_preference`: Morning (06-12), Afternoon (12-17), Evening (17-21), Night (21-24), or specific hours
  - `duration_minutes`: Integer activity duration in minutes
  - `health_profile`: `{ asthma: bool, senior: bool, child: bool, pregnancy: bool, cardiac_condition: bool, heat_sensitive: bool }`
  - `preferences`: `{ minimize_road_exposure: bool, prefer_green_areas: bool, avoid_industrial_zones: bool, prefer_water: bool }`
  - `clarifications_needed`: List of strings if the query lacks essential information (e.g. missing location).
- **Resilient Fallback Parser**: Include an AST/regex semantic parser to prevent system crashes during cold-starts or low-VRAM CPU environments, ensuring 100% uptime.

### 2. Live Data Services Layer (`app/data_services/`)
- **Geocoding (`geocoding_service.py`)**:
  - Live query to OpenStreetMap Nominatim: `https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1`
  - Required Header: Unique `User-Agent` (e.g., `TerraPulseAI-IITB-Hackathon/1.0`).
  - Cache results in memory using spatial TTL caching.
- **Atmospheric Weather Service (`weather_service.py`)**:
  - Query Open-Meteo: `https://api.open-meteo.com/v1/forecast`
  - Fetch 24-hour hourly variables: `temperature_2m`, `apparent_temperature`, `precipitation_probability`, `precipitation`, `wind_speed_10m`, `wind_direction_10m`, `wind_gusts_10m`, `cloud_cover`, `uv_index`, `visibility`.
- **Air Quality Service (`air_quality_service.py`)**:
  - Query Open-Meteo Air Quality: `https://air-quality-api.open-meteo.com/v1/air_quality`
  - Fetch current & hourly: `pm2_5`, `pm10`, `nitrogen_dioxide`, `ozone`, `european_aqi`.
- **Venue Discovery Service (`osm_service.py`)**:
  - Dynamically construct Overpass QL queries for candidate venues based on the specific activity:
    - *Running*: `node["leisure"="park"]`, `way["leisure"="track"]`, `way["highway"="pedestrian"]`
    - *Cycling*: `way["highway"="cycleway"]`, `way["highway"="track"]`, `way["leisure"="park"]`
    - *Golf*: `node["leisure"="golf_course"]`, `way["leisure"="golf_course"]`
    - *Cricket*: `node["sport"="cricket"]`, `way["leisure"="pitch"]["sport"="cricket"]`
    - *Football*: `node["sport"="soccer"]`, `way["leisure"="pitch"]`
    - *Tennis*: `node["sport"="tennis"]`, `way["leisure"="pitch"]`
    - *Hiking*: `way["highway"="path"]`, `way["leisure"="nature_reserve"]`, `node["tourism"="viewpoint"]`
    - *Picnic*: `node["leisure"="park"]`, `node["tourism"="picnic_site"]`, `node["leisure"="garden"]`
    - *Photography*: `node["tourism"="viewpoint"]`, `node["historic"="monument"]`, `node["tourism"="attraction"]`
    - *Outdoor Event*: `node["amenity"="marketplace"]`, `way["leisure"="park"]`
  - Fetch actual element IDs, names, coordinates, and geometries from `https://overpass-api.de/api/interpreter`.
- **Routing Service (`routing_service.py`)**:
  - Query OSRM routing engine `http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}` to calculate live driving/walking duration and distance.

### 3. Geospatial Vector Analysis Engine (`app/geospatial/`)
- **Road Exposure & Traffic Analysis (`road_analysis.py`)**:
  - Fetch highway vectors within 250m radius of venue via Overpass.
  - Classify arterial corridors (`motorway`, `trunk`, `primary`, `secondary`).
  - Compute Euclidean/Haversine distance from venue to nearest major road.
  - Calculate road surface density percentage: $(\sum \text{length} \times \text{width}) / (\pi \times r^2) \times 100$.
- **Green Canopy Cover (`green_cover.py`)**:
  - Fetch polygons within 300m for `leisure=park|garden|nature_reserve`, `landuse=grass|forest|meadow`, `natural=wood`.
  - Calculate vegetative fraction relative to total buffer area.
- **Industrial Proximity & Wind Vector Plume Advection (`industrial_analysis.py`)**:
  - Query industrial facilities within 3000m (`landuse=industrial`, `man_made=works`).
  - Calculate forward azimuth bearing $\theta_{\text{ind}}$ from industrial site to venue.
  - Compare with live wind direction $\theta_{\text{wind}}$ (degrees from which wind blows):
    $$\Delta\theta = |(\theta_{\text{wind}} - \theta_{\text{ind}} + 180) \pmod{360} - 180|$$
  - If $\Delta\theta < 45^\circ$, flag the venue as **downwind of an industrial plume** and apply severe air-quality safety penalties.

### 4. Deterministic Scoring & 24h Window Optimizer (`app/scoring/`)
- **Safety Constraints Matrix (`safety_constraints.py`)**:
  - Implement thresholds for all 10 activities: max/min temperature, max wind speed, precipitation probability, UV index, PM2.5.
  - Apply health adjustments:
    - *Asthma*: Cap PM2.5 at 45 µg/m³, wind speed at 30 km/h, min temperature at 8°C.
    - *Cardiac*: Cap temperature at 30°C, min temperature at 10°C, PM2.5 at 55 µg/m³.
    - *Senior*: Cap temperature at 32°C, UV index at 7.0, avoid steep slopes.
    - *Child*: Cap UV index at 7.0, limit high heat index.
    - *Pregnancy*: Cap temperature at 31°C, max PM2.5 at 50 µg/m³, limit intense exertion.
    - *Heat-Sensitive*: Cap temperature at 29°C, UV index at 6.5.
- **24-Hour Hourly Window Finder (`time_window_finder.py`)**:
  - Iterate through all 24 forecast hours.
  - Check safety bounds for contiguous blocks matching the user's duration.
  - Rank blocks by thermal comfort ($18^\circ\text{C} - 23^\circ\text{C}$ optimal), zero rain probability, and low UV glare.
- **Multi-Factor Ranking Formula (`ranking_engine.py`)**:
  $$\text{Score} = w_w S_{\text{weather}} + w_{aq} S_{\text{aqi}} + w_g S_{\text{geo}} + w_t S_{\text{travel}} + w_p S_{\text{pref}}$$
  where weights are dynamically calibrated per activity (e.g., Photography prioritizes cloud cover & geography; Running prioritizes weather & AQI).

### 5. Grounded Evidence Store & Provenance Registry (`app/evidence/`)
- **EvidenceRecord Model**:
  `{ id, claim, source, evidence_type, raw_value, unit, timestamp, method_description, reliability_score }`
- **Claim Validator (`validator.py`)**:
  Verifies that every generated statement mathematically matches an underlying `EvidenceRecord`.

### 6. Grounded SLM Explainer (`app/agent/explainer.py`)
- Feed the top venue, recommended time window, and verified evidence IDs into the <= 3B SLM.
- Generate concise, evidence-backed advice:
  - **What to carry**: (e.g. hydration pack for high heat index, windbreaker for 22 km/h wind, rain cover if precip > 30%).
  - **What to watch for**: (e.g. road crossing at 120m, dew on grass, UV index peaking at 8).
  - **When to avoid going**: (e.g. avoid after 11 AM due to heat spike and afternoon thunderstorm probability).

### 7. FastAPI Gateway (`app/main.py`)
- Endpoints:
  - `POST /api/v1/recommend`: Natural language query or structured recommendation request.
  - `GET /api/v1/venues`: Direct OSM venue discovery by coordinates and activity.
  - `GET /api/v1/activities`: Returns safety constraints and scoring criteria for all 10 activities.
  - `POST /api/v1/explain`: Grounded explanation for a selected venue and time slot.
  - `GET /api/v1/health`: Readiness probe testing upstream API reachability.

### 8. Frontend Dashboard (React + Leaflet + Rich Design System)
- **ActivityQueryBar**: Natural language search bar with 10 activity preset chips, duration slider, and health profile toggles.
- **InteractiveMap**: Leaflet map displaying:
  - Custom SVG pins for primary venue and alternatives.
  - 250m road exposure buffer circle (cyan dashed).
  - 300m green canopy buffer circle (emerald solid).
  - Industrial site location and wind vector arrow overlay.
- **TimelineWindowPicker**: 24-hour diurnal atmospheric strip displaying hourly temperature, UV index, and rain risk with the peak window highlighted.
- **RecommendationCard**: Circular score meter (0–100), factor breakdown progress bars, safety status pill (`✓ Verified Safe` or `⚠ Caution Advised`), and grounded reasons.
- **EvidenceAuditPanel**: Provenance drawer displaying all corroborated claims, data freshness timestamps, and calculation methods.

---

## III. VERIFICATION, DOCKER & SUBMISSION REQUIREMENTS

1. **Dockerized Deployment**:
   - Provide `docker-compose.yml` orchestrating FastAPI backend and React frontend.
2. **Automated Test Suite**:
   - Pytest suite covering intent parsing, safety constraints, asthma PM2.5 limits, multi-factor ranking, and API endpoints.
3. **Model Repository Artifact**:
   - Provide `upload_hf_model.py` and a complete `README_MODEL_CARD.md` with base model attribution, open license, and `from_pretrained` usage code ready for Hugging Face Hub submission.
