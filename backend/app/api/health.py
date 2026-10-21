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
async def health_check(request: Request, deep: bool = True):
    """Liveness + real upstream probes (cached for 60 s) + model status."""
    s = get_settings()
    state = request.app.state
    if deep and time.monotonic() - _cache["at"] > 60:
        http = state.http
        probes = await asyncio.gather(
            _probe(http, "open_meteo_forecast", "GET", f"{s.WEATHER_API_URL}/forecast",
                   params={"latitude": 19.07, "longitude": 72.88, "hourly": "temperature_2m", "forecast_days": 1}),
            _probe(http, "open_meteo_air_quality", "GET", f"{s.AIR_QUALITY_API_URL}/air-quality",
                   params={"latitude": 19.07, "longitude": 72.88, "hourly": "pm2_5", "forecast_days": 1}),
            _probe(http, "osrm", "GET", f"{s.OSRM_URL}/nearest/v1/driving/72.88,19.07"),
            _probe(http, "overpass", "POST", s.overpass_mirrors[0], data={"data": "[out:json][timeout:5];node(1);out;"}),
        )
        _cache.update(at=time.monotonic(), upstream=dict(probes))
    return {
        "status": "healthy",
        "service": s.API_TITLE,
        "version": s.API_VERSION,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": state.slm.describe(),
        "upstream": _cache["upstream"],
        "cache": {"entries": len(state.cache.store), "hits": state.cache.hits, "misses": state.cache.misses},
    }
