import asyncio
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.config import get_settings

router = APIRouter()
_cache = {"at": 0.0, "upstream": {}}


async def _probe(http, name: str, method: str, url: str, **kw) -> tuple:
    try:
        await http.request(method, url, source=name, retries=0, timeout_s=8, **kw)
        return name, "ok"
    except Exception as exc:  # noqa: BLE001
        return name, f"down: {str(exc)[:80]}"


@router.get("/health")
async def health_check(request: Request, deep: bool = True, probe: bool = False):
    """Liveness + real upstream probes (cached for 60 s) + model status."""
    s = get_settings()
    state = request.app.state
    if deep and time.monotonic() - _cache["at"] > 60:
        http = state.http
        probes = await asyncio.gather(
            _probe(http, "open_meteo_forecast", "GET", f"{s.WEATHER_API_URL}/forecast",
                   params={"latitude": s.HEALTH_PROBE_LAT, "longitude": s.HEALTH_PROBE_LON, "hourly": "temperature_2m", "forecast_days": 1}),
            _probe(http, "open_meteo_air_quality", "GET", f"{s.AIR_QUALITY_API_URL}/air-quality",
                   params={"latitude": s.HEALTH_PROBE_LAT, "longitude": s.HEALTH_PROBE_LON, "hourly": "pm2_5", "forecast_days": 1}),
            _probe(http, "osrm", "GET", f"{s.OSRM_URL}/nearest/v1/driving/{s.HEALTH_PROBE_LON},{s.HEALTH_PROBE_LAT}"),
            _probe(http, "overpass", "POST", s.overpass_mirrors[0], data={"data": "[out:json][timeout:5];node(1);out;"}),
        )
        _cache.update(at=time.monotonic(), upstream=dict(probes))
    slm_probe = None
    if probe and state.slm.ready:
        started = time.monotonic()
        try:
            text = await asyncio.wait_for(asyncio.to_thread(
                state.slm.complete, [{"role": "user", "content": "Say OK."}], max_tokens=5, temperature=0.0), 60)
            slm_probe = {"ok": True, "ms": round((time.monotonic() - started) * 1000), "text": text[:40]}
        except asyncio.TimeoutError:
            slm_probe = {"ok": False, "error": "no output within 60 s"}
        except Exception as exc:  # noqa: BLE001
            slm_probe = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    return {
        "status": "healthy",
        "slm_probe": slm_probe,
        "service": s.API_TITLE,
        "version": s.API_VERSION,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": state.slm.describe(),
        "upstream": _cache["upstream"],
        "cache": {"entries": len(state.cache.store), "hits": state.cache.hits, "misses": state.cache.misses},
    }
