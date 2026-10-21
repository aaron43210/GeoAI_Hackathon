"""End-to-end request pipeline, written as an async event stream (stage → … → final).

understand → clarify/scope → locate → candidates (OSM) → live data (weather, air, routes, map context) →
features & exposure → venue×hour grid → fact sheet → SLM answer → grounding check (repair once → template).
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from app.advice.template import body_of, headline, render as render_template
from app.agent.conversation import classify_turn, explain, smalltalk_reply
from app.agent.session import SessionStore
from app.agent.time_resolver import Resolution, TimeResolver
from app.agent.understand import understand
from app.config import get_settings
from app.data_services.air_quality_service import AirQualityService, WAQIService
from app.data_services.geocoding_service import GeocodingService, Place
from app.data_services.http import UpstreamError
from app.data_services.osm_service import OSMFeature, OSMService, haversine_m
from app.data_services.routing_service import RoutingService
from app.data_services.weather_service import WeatherService
from app.evidence.factsheet import FactSheetBuilder, fmt_window, osm_url
from app.evidence.grounding import verify
from app.geospatial.context import LocalProjection, VenueFeatures, area_m2, compute_features
from app.knowledge import get_knowledge
from app.scoring.grid import GridResult, VenueCtx, build_grid
from app.slm.prompts import REPAIR_NOTE, compose_messages
from app.slm.runtime import SLMRuntime, strip_think

logger = logging.getLogger(__name__)

SOURCES = ["Map data © OpenStreetMap contributors (ODbL) via Overpass API & Nominatim",
           "Weather & air quality: Open-Meteo.com (CC BY 4.0), air quality from Copernicus CAMS",
           "Routing: OSRM (Project OSRM demo server)"]
STANDARD_LIMITATIONS = [
    "air-quality forecast comes from a ~40 km CAMS grid, so differences between nearby places come from the modelled "
    "road/industry exposure, not from measurements",
    "travel times are free-flow OSRM times with a typical-traffic multiplier, not live traffic",
    "OpenStreetMap coverage varies by area",
]


class Pipeline:
    def __init__(self, geocoder: GeocodingService, osm: OSMService, weather: WeatherService,
                 air: AirQualityService, waqi: WAQIService, routing: RoutingService, slm: SLMRuntime,
                 sessions: SessionStore):
        self.settings = get_settings()
        self.kb = get_knowledge()
        self.geocoder, self.osm, self.weather, self.air, self.waqi, self.routing = \
            geocoder, osm, weather, air, waqi, routing
        self.slm, self.sessions = slm, sessions
        self.resolver = TimeResolver()

    # ------------------------------------------------------------------ public
    async def run(self, question: str, session_id: Optional[str] = None,
                  user_location: Optional[Tuple[float, float]] = None,
                  overrides: Optional[Dict[str, Any]] = None) -> AsyncIterator[Dict[str, Any]]:
        t0 = time.monotonic()
        timings: Dict[str, int] = {}
        overrides = overrides or {}

        def mark(name: str) -> None:
            timings[name] = round((time.monotonic() - t0) * 1000)

        prev = self.sessions.get(session_id)
        assumed_place_note: Optional[str] = None
        last = (prev or {}).get("last") or {}
        turn = classify_turn(question, bool(last.get("fact_sheet")))
        if turn != "plan":
            if turn == "explain":
                yield _stage("explain", "Looking at the data behind my last answer…")
                answer, grounding, path = await explain(question, last, self.slm)
                status, extra = "explanation", {"grounding": grounding, "answer_path": path,
                                                "evidence": last.get("evidence")}
            else:
                answer, status, extra = smalltalk_reply(turn), "chat", {}
            sid = self.sessions.save(session_id, history=_history(prev, question, answer))
            mark("total")
            yield {"event": "final", "data": {"status": status, "session_id": sid, "question": question,
                                              "answer_markdown": answer, "intent": {"path": turn},
                                              "model": self.slm.describe(), "timings_ms": timings,
                                              "sources": SOURCES, **extra}}
            return
        yield _stage("understand", "Reading your question…")
        intent, intent_path = await understand(question, self.slm, prev.get("intent") if prev else None)
        if prev:
            intent = _merge_follow_up(intent, prev)
        if overrides.get("profiles"):
            known = [p for p in overrides["profiles"] if p in self.kb.profiles]
            intent = type(intent)(**{**intent.model_dump(), "profiles": sorted(set(intent.profiles) | set(known))})
        mark("understand")

        def final(status: str, **kw: Any) -> Dict[str, Any]:
            saved: Dict[str, Any] = {"intent": intent.model_dump(), "user_location": user_location,
                                     "history": _history(prev, question, kw.get("answer_markdown", ""))}
            if kw.get("_place"):
                saved["place"] = kw["_place"]
            if kw.get("fact_sheet"):      # keep the last *recommendation* for "why?" follow-ups
                saved["last"] = {k: kw.get(k) for k in ("fact_sheet", "evidence", "answer_markdown", "grounding",
                                                         "limitations")}
            sid = self.sessions.save(session_id, **saved)
            kw.pop("_place", None)
            mark("total")
            return {"event": "final", "data": {
                "status": status, "session_id": sid, "question": question,
                "intent": {**intent.model_dump(), "path": intent_path},
                "model": self.slm.describe(), "timings_ms": timings, "sources": SOURCES, **kw}}

        if not intent.in_scope:
            acts = ", ".join(p.label.lower() for p in self.kb.activities.values())
            yield final("out_of_scope", answer_markdown=(
                "I can only help plan **outdoor activities** (where and when to go, what to carry, what to watch for). "
                f"I support: {acts}. Try e.g. *\"best time for a run near <your area> tomorrow?\"*"))
            return

        if not intent.activity:
            yield final("clarification_needed", clarification={
                "question": "Which outdoor activity are you planning?", "field": "activity",
                "options": [p.label for p in self.kb.activities.values()]},
                answer_markdown="Which outdoor activity are you planning? For example running, a walk, cycling, "
                                "cricket, golf, a picnic, a hike, photography or an outdoor event.")
            return

        # ------------------------------------------------------------ location
        place: Optional[Place] = None
        origin: Optional[Tuple[float, float]] = None
        if not intent.location_text and not user_location and not (prev and prev.get("place")):
            guessed = await self._tail_place(question)
            if guessed:
                intent = type(intent)(**{**intent.model_dump(), "location_text": guessed})
                assumed_place_note = f"I took “{guessed}” as the place; tell me if you meant somewhere else."
        if intent.location_text:
            yield _stage("locate", f"Finding “{intent.location_text}” on the map…")
            try:
                geo = await self.geocoder.geocode(intent.location_text)
            except UpstreamError as exc:
                yield final("error", answer_markdown="The geocoding service (Nominatim) is unavailable right now, so I "
                            "can't locate the place. Please try again in a minute.", errors=[str(exc)])
                return
            if not geo.best:
                yield final("clarification_needed", clarification={
                    "question": f"I couldn't find “{intent.location_text}”. Which area or city do you mean?",
                    "field": "location", "options": ["Use my location"]},
                    answer_markdown=f"I couldn't find **{intent.location_text}** on OpenStreetMap. Could you name a "
                                    "nearby area or city, or share your location?")
                return
            if geo.ambiguous and not overrides.get("accept_first_place"):
                opts = [p.display_name.split(",")[0] + ", " + (p.city or p.state or "") for p in geo.candidates[:3]]
                yield final("clarification_needed", clarification={
                    "question": f"There are several places called “{intent.location_text}”. Which one?",
                    "field": "location", "options": opts},
                    answer_markdown="Which one do you mean? " + " · ".join(f"**{o}**" for o in opts))
                return
            place = geo.best
            origin = user_location or (place.latitude, place.longitude)
        elif user_location:
            yield _stage("locate", "Using your location…")
            place = await self.geocoder.reverse(*user_location) or Place(
                name="your location", display_name="your location", latitude=user_location[0],
                longitude=user_location[1])
            origin = user_location
        elif prev and prev.get("place"):
            place = Place(**prev["place"])
            origin = prev.get("user_location") or (place.latitude, place.longitude)
        else:
            yield final("clarification_needed", clarification={
                "question": "Where should I look? Name an area or city, or share your location.",
                "field": "location", "options": ["Use my location"]},
                answer_markdown="Where should I look? Please name an area or city (e.g. a neighbourhood), or tap "
                                "**Use my location**.")
            return
        mark("locate")

        # ------------------------------------------------------------ time
        tz = ZoneInfo(self.settings.APP_TZ)
        now = datetime.now(tz).replace(tzinfo=None)
        tt = intent.time_text if intent.time_text and intent.time_text.lower() not in question.lower() else ""
        res: Resolution = self.resolver.resolve(f"{question} {tt}".strip(), now)
        notes = list(res.notes)
        if assumed_place_note:
            notes.append(assumed_place_note)
        horizon = self.resolver.horizon_check(res, now)
        if "beyond_weather_horizon" in horizon or not res.windows:
            yield final("not_available", _place=place.to_dict(), answer_markdown=(
                "That's beyond the range of the live forecast (about 15 days), so I can't give a grounded answer yet. "
                "Ask me again closer to the date." if res.windows else
                "That time has already passed. Tell me another day or time and I'll check it."),
                notes=notes)
            return
        act = self.kb.activities[intent.activity]
        duration = int(overrides.get("duration_min") or intent.duration_min or act.default_duration_min)
        max_travel = float(overrides.get("max_travel_min") or intent.max_travel_min or self.settings.DEFAULT_MAX_TRAVEL_MIN)
        radius = float(overrides.get("radius_km") or intent.radius_km or act.default_radius_km)
        radius = min(radius, self.settings.MAX_RADIUS_KM)
        if not intent.radius_km and not overrides.get("radius_km"):
            notes.append(f"No search radius given: searched within {radius:g} km.")

        # ------------------------------------------------------------ candidates
        yield _stage("venues", f"Looking for {act.label.lower()} places within {radius:g} km of {place.name}…")
        limitations = list(STANDARD_LIMITATIONS)
        venues_feat: List[OSMFeature] = []
        osm_ok = True
        try:
            cand = await self.osm.find_candidates(act.id, place.latitude, place.longitude, radius)
            venues_feat = self._select(cand.candidates, place, radius, act)
            if not venues_feat and radius < self.settings.MAX_RADIUS_KM:
                bigger = min(self.settings.MAX_RADIUS_KM, radius * self.kb.geo["candidates"]["radius_expand_factor"])
                yield _stage("venues", f"Nothing mapped within {radius:g} km, widening to {bigger:g} km…")
                cand = await self.osm.find_candidates(act.id, place.latitude, place.longitude, bigger)
                venues_feat = self._select(cand.candidates, place, bigger, act)
                if venues_feat:
                    notes.append(f"Nothing suitable was mapped within {radius:g} km, so I widened the search to {bigger:g} km.")
                radius = bigger
        except UpstreamError as exc:
            logger.warning("Overpass unavailable (%s); trying Nominatim category search", exc)
            yield _stage("venues", "Map server busy, searching OpenStreetMap via Nominatim instead…")
            venues_feat = await self._nominatim_candidates(act, place, radius)
            if venues_feat:
                limitations.insert(0, "venues came from OpenStreetMap via Nominatim because Overpass did not respond")
            else:
                osm_ok = False
                limitations.insert(0, "OpenStreetMap venue data was unavailable (Overpass and Nominatim did not "
                                      "return venues), so only conditions for the area itself are shown")
        if osm_ok and not venues_feat:
            notes.append(f"No {act.label.lower()} venues are mapped in OpenStreetMap within {radius:g} km; showing "
                         "conditions for the area itself.")
        mark("venues")

        # ------------------------------------------------------------ live data
        yield _stage("data", f"Fetching live weather, air quality and routes for {max(1, len(venues_feat))} place(s)…")
        if venues_feat:
            points = [(v.lat, v.lon) for v in venues_feat]
        else:
            points = [(place.latitude, place.longitude)]
        days = sorted({w.day for w in res.windows})
        start, end = days[0] - timedelta(days=1), days[-1]
        aq_end_ok = (end - now.date()).days <= self.kb.time["horizon_days"]["air_quality"]
        wx_t = asyncio.create_task(self.weather.hourly(points, start, end))
        aq_t = asyncio.create_task(self.air.hourly(points, start, end)) if aq_end_ok else None
        rt_t = asyncio.create_task(self.routing.table(origin, points)) if venues_feat else None
        ctx_t = asyncio.create_task(self.osm.fetch_context(venues_feat)) if venues_feat else None
        waqi_t = asyncio.create_task(self.waqi.nearest(place.latitude, place.longitude))
        try:
            wx = await wx_t
        except UpstreamError as exc:
            for t in (aq_t, rt_t, ctx_t, waqi_t):
                if t:
                    t.cancel()
            yield final("error", _place=place.to_dict(), errors=[str(exc)], answer_markdown=(
                "The live weather forecast (Open-Meteo) is unavailable right now. I won't recommend a time without "
                "it. Please try again shortly."))
            return
        aq = await _soft(aq_t, limitations, "air-quality forecast was unavailable, so air quality was not checked")
        if not aq_end_ok:
            limitations.insert(0, "the air-quality forecast only covers the next ~4 days, so air quality was not checked")
        routes = await _soft(rt_t, limitations, "travel times were unavailable (OSRM did not respond)")
        ctx = await _soft(ctx_t, limitations, "map context (roads, industry, green, water) was unavailable")
        waqi = await _soft(waqi_t, None, None)
        mark("data")

        # sunrise/sunset anchored windows ("sunset photos") from the live daily forecast
        sun = {}
        for d, sr, ss in zip(wx[0].daily.get("time", []), wx[0].daily.get("sunrise", []), wx[0].daily.get("sunset", [])):
            a, b = datetime.fromisoformat(sr), datetime.fromisoformat(ss)
            sun[datetime.fromisoformat(d).date()] = (a.hour + a.minute / 60, b.hour + b.minute / 60)
        windows = self.resolver.apply_anchors(res.windows, sun)

        # ------------------------------------------------------------ analysis
        yield _stage("score", f"Scoring {max(1, len(venues_feat))} place(s) across every hour of your window…")
        ctxs: List[VenueCtx] = []
        if venues_feat:
            for i, v in enumerate(venues_feat):
                feats = compute_features(v, ctx) if ctx is not None else VenueFeatures.unavailable()
                travel = routes[i] if routes else None
                ctxs.append(VenueCtx(id=v.osm_id, name=v.name, lat=v.lat, lon=v.lon, kind=v.kind, tags=v.tags,
                                     osm_url=osm_url(v.osm_id), features=feats, wx=wx[i], aq=aq[i] if aq else None,
                                     travel=travel, straight_km=haversine_m(origin[0], origin[1], v.lat, v.lon) / 1000))
        else:
            ctxs.append(VenueCtx(id="area", name=f"{place.name} (area, no specific venue)", lat=place.latitude,
                                 lon=place.longitude, kind="area", tags={}, osm_url=place.source_url,
                                 features=VenueFeatures.unavailable(), wx=wx[0], aq=aq[0] if aq else None))
        grid = await asyncio.to_thread(build_grid, ctxs, windows, act, intent.profiles, intent.preferences,
                                       duration, max_travel, now)
        mark("score")

        builder = FactSheetBuilder(grid, act, intent.profiles, question, place.display_name.split(",")[0]
                                   if place.display_name else place.name, duration, notes, limitations, waqi)
        sheet = builder.build()

        # ------------------------------------------------------------ compose + verify
        yield _stage("compose", "Writing the recommendation from the fetched facts…")
        must = ctxs[(grid.best or grid.least_bad).venue].name if (grid.best or grid.least_bad) else None
        answer, grounding, path = None, None, "template"
        if self.slm.ready and (grid.best or grid.least_bad):
            draft = ""
            async for tok in _stream(self.slm, compose_messages(sheet.text), self.settings.SLM_MAX_TOKENS_ANSWER):
                draft += tok
                yield {"event": "token", "data": {"t": tok}}
            draft = strip_think(draft)
            head = headline(sheet.text)
            if draft.strip():
                draft = f"{head}\n\n{body_of(draft)}" if head else draft   # the decision line is code-rendered
            g1 = verify(draft, sheet.text, sheet.ledger.ids(), sheet.allowed_names, must, recommended=grid.best is not None,
                        ledger_values={e.id: e.value for e in sheet.ledger.items})
            if not draft.strip():
                # the model produced nothing (e.g. GPU quota exhausted): don't retry, use the data template
                grounding = {"status": "templated", "reason": "language model unavailable for this request"}
            elif g1.status == "passed":
                answer, grounding, path = draft, {"status": "passed", **_g(g1)}, "slm"
            else:
                yield _stage("verify", "Fixing facts that didn't match the data…")
                msgs = compose_messages(sheet.text) + [{"role": "assistant", "content": draft},
                                                       {"role": "user", "content": REPAIR_NOTE.format(
                                                           violations="; ".join(g1.violations))}]
                try:
                    fixed = strip_think(await asyncio.to_thread(self.slm.complete, msgs,
                                                                max_tokens=self.settings.SLM_MAX_TOKENS_ANSWER))
                except Exception as exc:  # noqa: BLE001 - e.g. GPU quota; never fail the request over phrasing
                    logger.warning("SLM repair failed (%s); using the data template", exc)
                    fixed = ""
                if fixed.strip() and head:
                    fixed = f"{head}\n\n{body_of(fixed)}"
                g2 = verify(fixed, sheet.text, sheet.ledger.ids(), sheet.allowed_names, must, recommended=grid.best is not None,
                        ledger_values={e.id: e.value for e in sheet.ledger.items})
                if g2.status == "passed":
                    answer, grounding, path = fixed, {"status": "repaired", "first_violations": g1.violations,
                                                      **_g(g2)}, "slm (repaired)"
                else:
                    grounding = {"status": "templated", "first_violations": g1.violations,
                                 "repair_violations": g2.violations}
        if answer is None:
            answer = render_template(sheet.text)
            grounding = grounding or {"status": "templated", "reason": "SLM unavailable" if not self.slm.ready else
                                      "no recommendation to phrase"}
        mark("compose")

        status = "ok" if grid.best else "not_recommended"
        if (not osm_ok or aq is None or routes is None or (venues_feat and ctx is None)) and grid.best:
            status = "partial"
        yield final(status, _place=place.to_dict(), answer_markdown=answer, answer_path=path, grounding=grounding,
                    fact_sheet=sheet.text, evidence=sheet.ledger.to_list(), notes=notes, limitations=limitations,
                    place=place.to_dict(), origin={"lat": origin[0], "lon": origin[1]},
                    resolved_windows=[w.to_dict() for w in windows], **_structured(grid, sheet, ctxs))

    # ------------------------------------------------------------------ helpers
    async def _tail_place(self, question: str) -> Optional[str]:
        """A trailing place name with no spatial cue ("… kal shaam dwarka delhi"), accepted only if the live
        geocoder resolves it to a settlement type from geo.yaml (never a shop/POI that happens to share the words)."""
        from app.agent.rule_parser import tail_place_candidates
        settlement = set(self.kb.geo["place_type_priority"])
        for cand in tail_place_candidates(question)[:2]:
            try:
                geo = await self.geocoder.geocode(cand)
            except UpstreamError:
                return None
            if geo.best and geo.best.osm_type in settlement:
                return cand
        return None

    async def _nominatim_candidates(self, act, place: Place, radius_km: float) -> List[OSMFeature]:
        feats: List[OSMFeature] = []
        for phrase in act.nominatim_phrases:
            try:
                rows = await self.geocoder.search_category(phrase, place.latitude, place.longitude, radius_km)
            except UpstreamError:
                continue
            for r in rows:
                bb = r.get("boundingbox")
                tags = {**(r.get("extratags") or {}), r.get("category", "place"): r.get("type", "yes")}
                if r.get("name"):
                    tags["name"] = r["name"]
                feats.append(OSMFeature(osm_id=f"{r.get('osm_type')}/{r.get('osm_id')}", tags=tags,
                                        lat=float(r["lat"]), lon=float(r["lon"]), kind=r.get("osm_type", "node"),
                                        bounds={"minlat": float(bb[0]), "maxlat": float(bb[1]), "minlon": float(bb[2]),
                                                "maxlon": float(bb[3])} if bb else None))
        return self._select(feats, place, radius_km, act)

    def _select(self, feats: List[OSMFeature], place: Place, radius_km: float, act) -> List[OSMFeature]:
        cfg = self.kb.geo["candidates"]
        excluded = set(cfg["excluded_access"])
        synonyms = {s.lower() for s in act.synonyms} | {act.id}
        scored = []
        for f in feats:
            if not f.name or len(f.name.strip()) < cfg["min_name_chars"] or f.tags.get("access") in excluded:
                continue
            sport_tag = f.tags.get("sport", "").lower()
            if act.require_sport_match and sport_tag and not any(s in sport_tag for s in synonyms | {act.id}):
                continue
            d_km = haversine_m(place.latitude, place.longitude, f.lat, f.lon) / 1000
            if d_km > radius_km * cfg["radius_slack"]:
                continue
            a = area_m2(f, LocalProjection(f.lat, f.lon)) if f.kind != "node" else None
            if a is not None and act.min_area_m2 and a < act.min_area_m2:
                continue
            sport = f.tags.get("sport", "").lower()
            tag_hit = 1.0 if sport and any(s in sport for s in synonyms) else 0.0
            area_term = min(1.0, math.log10(a) / cfg["area_log_scale"]) if a and a > 1 else 0.0
            pre = -cfg["distance_weight"] * d_km / radius_km + cfg["area_weight"] * area_term + cfg["tag_weight"] * tag_hit
            scored.append((pre, f))
        scored.sort(key=lambda x: -x[0])
        chosen: List[OSMFeature] = []
        for _, f in scored[:cfg["shortlist_pool"]]:
            if any(f.name.lower() == c.name.lower() or haversine_m(f.lat, f.lon, c.lat, c.lon) < cfg["dedupe_m"]
                   for c in chosen):
                continue
            chosen.append(f)
            if len(chosen) >= cfg["max_candidates"]:
                break
        return chosen


def _history(prev: Optional[Dict[str, Any]], question: str, answer: str) -> List[Dict[str, str]]:
    turns = list((prev or {}).get("history") or [])
    turns += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    keep = get_knowledge().language["conversation"]["history_turns"] * 2
    return turns[-keep:]


def _stage(name: str, message: str) -> Dict[str, Any]:
    return {"event": "stage", "data": {"name": name, "message": message}}


def _g(r) -> Dict[str, Any]:
    return {"facts_checked": r.facts_checked, "violations": r.violations}


async def _soft(task, limitations: Optional[List[str]], note: Optional[str]):
    if task is None:
        return None
    try:
        return await task
    except (UpstreamError, asyncio.CancelledError) as exc:
        logger.warning("optional source failed: %s", exc)
        if limitations is not None and note:
            limitations.insert(0, note)
        return None


async def _stream(slm: SLMRuntime, messages, max_tokens: int) -> AsyncIterator[str]:
    """Bridge the blocking llama.cpp token stream into the event loop."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    def worker():
        try:
            for tok in slm.stream(messages, max_tokens=max_tokens):
                loop.call_soon_threadsafe(queue.put_nowait, tok)
        except Exception as exc:  # noqa: BLE001
            logger.error("SLM stream failed: %s", exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    fut = loop.run_in_executor(None, worker)
    while True:
        item = await queue.get()
        if item is sentinel:
            break
        yield item
    await fut


def _merge_follow_up(intent, prev: Dict[str, Any]):
    p = prev.get("intent") or {}
    data = intent.model_dump()
    if not data.get("activity") and p.get("activity"):
        data["activity"] = p["activity"]
        data["in_scope"] = True
    if data.get("is_follow_up") or not data.get("activity") or data["activity"] == p.get("activity"):
        for k in ("radius_km", "max_travel_min", "duration_min", "age"):
            if data.get(k) is None and p.get(k) is not None:
                data[k] = p[k]
        data["profiles"] = sorted(set(data.get("profiles", [])) | set(p.get("profiles", [])))
    return type(intent)(**data)


def _structured(grid: GridResult, sheet, ctxs: List[VenueCtx]) -> Dict[str, Any]:
    """Machine-readable view for the UI (map, heatmap, cards)."""
    def pick(p):
        if p is None:
            return None
        v = ctxs[p.venue]
        return {"venue": {"id": v.id, "name": v.name, "lat": v.lat, "lon": v.lon, "kind": v.kind,
                          "osm_url": v.osm_url, "type": next((v.tags[k] for k in ("leisure", "sport", "tourism",
                                                               "natural", "historic") if k in v.tags), v.kind)},
                "start": p.start.isoformat(), "end": p.end.isoformat(), "label": fmt_window(p), "score": p.score,
                "status": p.status, "concerns": [h.text for h in p.concerns], "unknowns": p.unknowns,
                "travel": v.travel, "travel_at_start_min": p.cells[0].travel_min,
                "factors": {f: round(sum(c.factors[f] for c in p.cells if c.factors.get(f) is not None) /
                                     max(1, sum(1 for c in p.cells if c.factors.get(f) is not None)), 3)
                            for f in grid.weights if any(c.factors.get(f) is not None for c in p.cells)}}

    hours = sorted({c.ts for row in grid.cells.values() for c in row})
    heat = {"hours": hours, "venues": [v.name for v in ctxs], "scores": [], "status": [], "reasons": [],
            "wind": []}
    for vi in range(len(ctxs)):
        by_ts = {c.ts: c for c in grid.cells.get(vi, [])}
        heat["scores"].append([by_ts[t].score if t in by_ts else None for t in hours])
        heat["status"].append([by_ts[t].status if t in by_ts else None for t in hours])
        heat["reasons"].append([[h.text for h in by_ts[t].hits if h.state.value == "true"] if t in by_ts else []
                                for t in hours])
    first = {c.ts: c for c in grid.cells.get(0, [])}
    heat["wind"] = [{"from_deg": first[t].vars.get("wind_direction_10m"), "kmh": first[t].vars.get("wind_speed_10m")}
                    if t in first else None for t in hours]
    focus = grid.best or grid.least_bad
    timeline = []
    if focus is not None:
        for c in grid.cells.get(focus.venue, []):
            timeline.append({"ts": c.ts, "apparent": c.vars.get("apparent_temperature"),
                             "rain_prob": c.vars.get("precipitation_probability"), "uv": c.vars.get("uv_index"),
                             "us_aqi": c.vars.get("us_aqi"), "wind_kmh": c.vars.get("wind_speed_10m"),
                             "wind_from": c.vars.get("wind_direction_10m"),
                             "exposure": c.exposure.index if c.exposure else None, "score": c.score,
                             "status": c.status,
                             "in_best": bool(grid.best and grid.best.start <= c.dt < grid.best.end)})
    features = []
    for v in ctxs:
        f = v.features
        features.append({"name": v.name, "lat": v.lat, "lon": v.lon, "available": f.available,
                         "green_pct": round(f.green_fraction * 100) if f.available else None,
                         "nearest_major_road_m": f.nearest_major_road_m, "nearest_major_road": f.nearest_major_road_name,
                         "nearest_water_m": f.nearest_water_m, "nearest_water": f.nearest_water_name,
                         "industry": f.industry_within_3km[:3], "travel": v.travel})
    return {"recommendation": pick(grid.best), "least_bad": pick(grid.least_bad),
            "alternatives": [pick(a) for a in grid.alternatives], "alt_time": pick(grid.alt_time),
            "compare": [{"day": d["day"].isoformat(), "best": pick(d["best"]), "gated_windows": d["gated_hours"]}
                        for d in grid.compare],
            "avoid": sheet.avoid, "advice": sheet.advice, "heatmap": heat, "venues": features, "timeline": timeline,
            "limits": grid.limits}
