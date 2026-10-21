# MEMORY.md: project memory (living log)

Short-term memory for whoever works next (human or AI agent). Update at the end of every session: Progress,
Decisions, Findings, Open questions, Session log. Newest first. Keep it under ~200 lines.

## Snapshot (21 Oct 2026, evening IST)
- **Deadline:** Fri 24 Oct 2026, 23:59 IST · internal freeze 18:00 IST · submit by 21:00 IST.
- **Code:** backend v2 + new frontend implemented and working end-to-end locally with base Qwen3-1.7B (Q4_K_M).
  45 offline tests green. Live eval: 18/20 expected statuses (the 2 misses are legitimate clarifications), activity
  100%, profiles 100%, SLM-written answers 85% (rest template), p50 15.5 s.
- **HF user:** `aaronr43210` · repos not yet created.
- **Live endpoint:** not deployed yet.
- **Backup of the original v1 code:** `~/Desktop/GEOAI_v1_backup_2026-10-21.tar.gz` (the folder is not a git repo).

## Progress
- [x] Audit of v1 (fabricated venues/weather/AQI, wrong AQ URL, Bengaluru default, Phi-3 3.8B) → replaced
- [x] Knowledge base in YAML (activities ×13, profiles ×7, rules, advice, time lexicon, geo, scoring, grounding)
- [x] Data layer: HttpGateway (rate limits, backoff), geohash TTL cache, Nominatim, Overpass race + breaker,
      Nominatim category fallback, Open-Meteo multi-coord weather + AQ (with sub-indices), OSRM table, WAQI optional
- [x] Engine: time resolver, rule parser, SLM intent, grid, exposure, advice, ledger, fact sheet, verifier, template
- [x] API: /ask (+GET, ?format=text), /ask/stream (SSE), /recommend, /explain, /venues, /activities, /health
- [x] Frontend: chat + SSE, clarification chips, profile toggles, map (score colours, buffers, industry, wind),
      place×time heatmap with scrubber, hour-by-hour strip, factor bars, evidence drawer
- [x] Training assets: gen_intent_data.py, build_compose_data.py, 2 Colab notebooks, MODEL_CARD.md, upload_hf_model.py
- [x] Dockerfile (single container, port 7860), docker-compose, HF Space README, README rewrite
- [ ] `git init` + push to GitHub (needed for the Colab notebooks' REPO_URL, or upload a zip)
- [ ] Run `training/upload_hf_model.py --mode base` (creates the config repo now) ← quick win
- [ ] Colab: notebook 01 (data) → 02 (LoRA) → push adapter/merged/GGUF; set `SLM_REPO/SLM_FILE` (defaults already match)
- [ ] Deploy (HF Docker Space or Oracle Always Free); UptimeRobot on /api/v1/health; fill URL in README
- [ ] Slides (5): problem, architecture, worked example, model + eval table, limitations
- [ ] Final live eval + demo rehearsal + screen-recording backup

## Decisions (with why)
- **2026-10-21: Qwen3-1.7B chosen by a measured bake-off** (`backend/eval/bakeoff.py`, CPU Q4_K_M, 24 gold questions):
  activity 91.7 / location 100 / profiles 95.8 / scope 91.7, 2.2 s per intent, 0 invented numbers. Qwen3.5-2B was
  87.5 / 95.8 / 95.8 / 95.8 at 4.9 s and asserted an unsupported claim. Other candidates excluded on size or licence.
- **2026-10-21: SLM = language only; numbers/dates/safety deterministic; verifier enforces grounding.** Pattern
  validated by similar open projects (Weather-Agent: YAML policies + validator; TrailMate: picks checked against OSM).
- **2026-10-21: Three-valued rules** (TRUE/FALSE/UNKNOWN). Missing data is reported as unverified, never treated as safe.
- **2026-10-21: AQI thresholds follow US EPA activity guidance:** everyone: caution >100, strong caution >150,
  avoid >200; sensitive profiles: avoid >150, caution >100 (asthma/pregnancy/cardiac from 50).
- **2026-10-21: No preloaded datasets** (user spec). Geofabrik fallback dropped; Nominatim category search is the live
  fallback for venues.
- **2026-10-21: Training only in Colab** (user instruction). Teacher = open Qwen3-8B; drafts kept only if they pass the
  runtime verifier.
- **2026-10-21: Single container** serving UI + API (port 7860 in Docker, 8000 locally).
- **2026-10-21: Clarify genuinely ambiguous places** (equal-rank same-type places in different states, e.g. Jayanagar);
  otherwise prefer the bigger/urban settlement type.

## Findings (verified, don't re-investigate)
- Open-Meteo AQ path is `/v1/air-quality` (hyphen). Forecast + AQ accept comma-separated multi-coordinates (JSON list).
- CAMS US AQI over Mumbai is often ozone-driven (e.g. 280 with PM2.5 sub-index 167); we report the dominant pollutant.
- From this Mac/network, overpass-api.de, kumi.systems and private.coffee are unreachable even outside the sandbox;
  maps.mail.ru works but is slow and returns 504s under load. Expect better from a cloud host; the breaker plus
  Nominatim fallback keep answers flowing.
- Nominatim `q=<category>&viewbox=…&bounded=1` returns real OSM features (parks, golf courses, beaches, viewpoints).
- Overpass `out` geometry modes (`geom`/`bb`/`center`) are mutually exclusive; candidates use `out tags bb`, context
  uses a `(...)->.c; ... (around.c:R)` set query.
- `unsloth/Qwen3-1.7B-GGUF` has Q4_K_M (Qwen's own repo only Q8_0). Unsloth also has Qwen3.5-2B GGUFs.
- llama-cpp-python 0.3.36 installs on Python 3.14 (macOS) and runs Qwen3 and Qwen3.5 GGUFs.
- Example live answers: Bengaluru seniors walk → Raj Bhavan Gardens 07:00–08:00, avoid 13–21 (81% rain, storms), 10 s.

## Open questions (need the user)
- Team name and team size? Which other AI coding tools to disclose?
- Can `aaronr43210` create a free CPU-basic **Docker** Space? If not, OK to use Oracle Cloud Always Free?
- WAQI token (optional) for ground-station cross-checks?
- GitHub repo URL (for the notebooks and the submission form)?

## Session log
- 2026-10-21 (2) · Researched similar repos, rebuilt the backend (knowledge YAML, live-only data, grid, verifier),
  model bake-off, frontend rewrite, tests, training notebooks, Docker, README; live-tested many queries.
- 2026-10-21 (1) · Audited repo, probed APIs, wrote plan + CONNECTIONS/AGENTS/MEMORY/CLAUDE.md.
