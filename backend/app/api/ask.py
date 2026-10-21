import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


class LatLon(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=600, description="Free-form question")
    session_id: Optional[str] = Field(None, description="Pass back the session_id from a previous answer for follow-ups")
    user_location: Optional[LatLon] = Field(None, description="Browser geolocation, used for 'near me'")
    overrides: Dict[str, Any] = Field(default_factory=dict,
                                      description="Optional radius_km, max_travel_min, duration_min, accept_first_place")

    @field_validator("question")
    @classmethod
    def strip(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question cannot be empty")
        return v.strip()


async def _collect(request: Request, body: AskRequest) -> Dict[str, Any]:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(503, "pipeline not initialised")
    loc = (body.user_location.lat, body.user_location.lon) if body.user_location else None

    async def go():
        async for ev in pipeline.run(body.question, body.session_id, loc, body.overrides):
            if ev["event"] == "final":
                return ev["data"]
        raise HTTPException(500, "pipeline ended without a result")

    try:
        return await asyncio.wait_for(go(), timeout=get_settings().REQUEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(504, "timed out while fetching live data; please retry")


def _as_text(result: Dict[str, Any]) -> str:
    """Markdown answer + score breakdown + data timestamps + sources (for humans and curl)."""
    from app.scoring.grid import FACTOR_LABELS
    parts = [result.get("answer_markdown", "")]
    pick = result.get("recommendation") or result.get("least_bad")
    if pick and pick.get("factors"):
        bits = ", ".join(f"{FACTOR_LABELS.get(k, k)} {round(v * 100)}%" for k, v in pick["factors"].items())
        parts.append(f"Score {pick['score']}/100 = weighted sub-scores (activity weights in knowledge/activities.yaml): "
                     f"{bits}" + (f"; status: {pick['status']}" if pick.get("status") else ""))
    times = sorted({e["retrieved_at"] for e in result.get("evidence", []) if e.get("retrieved_at")})
    if times:
        parts.append(f"Live data retrieved {times[0].replace('T', ' ')[:16]} UTC; travel times are OSRM free-flow "
                     "with a typical-traffic estimate (no live traffic).")
    return "\n\n".join(parts) + "\n\n---\n" + "\n".join(result.get("sources", []))


@router.post("/ask")
async def ask(body: AskRequest, request: Request, format: str = Query("json", pattern="^(json|text)$")):
    """Ask the assistant. Returns the grounded answer, structured recommendation, evidence and sources.
    `?format=text` returns only the markdown answer (handy for curl)."""
    result = await _collect(request, body)
    if format == "text" or "text/plain" in request.headers.get("accept", ""):
        return PlainTextResponse(_as_text(result))
    return result


@router.get("/ask")
async def ask_get(request: Request, q: str = Query(..., min_length=2, max_length=600),
                  session_id: Optional[str] = None, format: str = Query("json", pattern="^(json|text)$")):
    """GET convenience: /api/v1/ask?q=...&format=text"""
    return await ask(AskRequest(question=q, session_id=session_id), request, format)


@router.get("/ask/stream")
async def ask_stream(request: Request, q: str = Query(..., min_length=2, max_length=600),
                     session_id: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None,
                     profiles: Optional[str] = Query(None, description="comma-separated profile ids from /activities")):
    """Server-sent events: `stage` progress, `token` draft tokens, then `final` (same body as POST /ask)."""
    pipeline = request.app.state.pipeline
    loc = (lat, lon) if lat is not None and lon is not None else None
    overrides = {"profiles": [p.strip() for p in profiles.split(",") if p.strip()]} if profiles else {}

    async def events():
        try:
            async for ev in pipeline.run(q, session_id, loc, overrides):
                if await request.is_disconnected():
                    break
                yield {"event": ev["event"], "data": json.dumps(ev["data"], default=str, ensure_ascii=False)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("stream failed")
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}

    return EventSourceResponse(events())


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=600)
    session_id: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    radius_km: Optional[float] = Field(None, gt=0, le=50)
    duration_minutes: Optional[int] = Field(None, ge=15, le=600)
    profiles: list[str] = Field(default_factory=list)


@router.post("/recommend")
async def recommend(body: RecommendRequest, request: Request):
    """Alias of /ask that accepts `query` and optional structured overrides."""
    loc = LatLon(lat=body.latitude, lon=body.longitude) if body.latitude is not None and body.longitude is not None else None
    overrides = {k: v for k, v in {"radius_km": body.radius_km, "duration_min": body.duration_minutes,
                                    "profiles": body.profiles}.items() if v}
    return await _collect(request, AskRequest(question=body.query, session_id=body.session_id, user_location=loc,
                                              overrides=overrides))


@router.get("/explain")
async def explain(request: Request, session_id: str):
    """The fact sheet, evidence ledger and grounding report behind a session's last answer."""
    item = request.app.state.pipeline.sessions.get(session_id)
    if not item or not item.get("last") or not item["last"].get("fact_sheet"):
        raise HTTPException(404, "no answer with evidence for this session")
    return item["last"]


@router.get("/venues")
async def venues(request: Request, activity: str, place: Optional[str] = None, lat: Optional[float] = None,
                 lon: Optional[float] = None, radius_km: float = Query(10, gt=0, le=50)):
    """Live OSM venue discovery for an activity (Overpass, falling back to Nominatim category search)."""
    from app.data_services.geocoding_service import Place
    from app.data_services.http import UpstreamError
    from app.knowledge import get_knowledge
    pipe = request.app.state.pipeline
    kb = get_knowledge()
    if activity not in kb.activities:
        raise HTTPException(422, f"activity must be one of {list(kb.activities)}")
    if lat is None or lon is None:
        if not place:
            raise HTTPException(422, "give place or lat/lon")
        geo = await pipe.geocoder.geocode(place)
        if not geo.best:
            raise HTTPException(404, f"could not find {place}")
        p = geo.best
    else:
        p = Place(name="point", display_name="point", latitude=lat, longitude=lon)
    act = kb.activities[activity]
    try:
        cand = await pipe.osm.find_candidates(activity, p.latitude, p.longitude, radius_km)
        chosen, source = pipe._select(cand.candidates, p, radius_km, act), "OpenStreetMap via Overpass"
    except UpstreamError:
        chosen, source = await pipe._nominatim_candidates(act, p, radius_km), "OpenStreetMap via Nominatim"
    return {"centre": {"lat": p.latitude, "lon": p.longitude, "name": p.name}, "source": source,
            "venues": [{"name": v.name, "lat": v.lat, "lon": v.lon, "osm_id": v.osm_id,
                        "osm_url": f"https://www.openstreetmap.org/{v.osm_id}", "tags": v.tags} for v in chosen]}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=600, description="What the user says")
    session_id: Optional[str] = Field(None, description="Send back the session_id you received to continue a chat")
    user_location: Optional[LatLon] = None


@router.post("/chat")
async def chat(body: ChatRequest, request: Request):
    """Conversational endpoint: plans, follow-ups ("what about Sunday?", "somewhere closer?"), "why?" questions
    about the last answer, greetings and help. Returns a compact reply; GET /explain gives the full evidence."""
    result = await _collect(request, AskRequest(question=body.message, session_id=body.session_id,
                                                user_location=body.user_location))
    pick = result.get("recommendation") or result.get("least_bad")
    return {
        "session_id": result.get("session_id"),
        "status": result.get("status"),
        "reply": result.get("answer_markdown", ""),
        "clarification": result.get("clarification"),
        "recommendation": {k: pick.get(k) for k in ("venue", "label", "score", "status", "concerns")} if pick else None,
        "grounding": result.get("grounding"),
        "understood_as": {k: (result.get("intent") or {}).get(k) for k in
                          ("activity", "location_text", "time_text", "profiles", "path")},
    }
