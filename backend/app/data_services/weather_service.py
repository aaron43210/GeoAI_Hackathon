"""Open-Meteo hourly forecast (+ daily sunrise/sunset) for many points in a single request."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import get_settings
from app.data_services.cache import Cache, geohash
from app.data_services.http import HttpGateway, UpstreamError

logger = logging.getLogger(__name__)

HOURLY_VARS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation_probability",
    "precipitation", "weather_code", "cloud_cover", "visibility", "wind_speed_10m", "wind_direction_10m",
    "wind_gusts_10m", "uv_index", "is_day",
]
DAILY_VARS = ["sunrise", "sunset", "uv_index_max", "precipitation_sum"]


@dataclass
class PointSeries:
    """Hourly series for one point. `hourly[var][i]` aligns with `times[i]` (local ISO, no offset)."""
    lat: float
    lon: float
    grid_lat: float
    grid_lon: float
    times: List[str]
    hourly: Dict[str, List[Any]]
    daily: Dict[str, List[Any]] = field(default_factory=dict)
    timezone: str = "UTC"
    utc_offset_seconds: int = 0
    source: str = ""
    url: str = ""
    retrieved_at: str = ""

    def value(self, var: str, ts: str) -> Optional[Any]:
        try:
            idx = self.times.index(ts)
        except ValueError:
            return None
        series = self.hourly.get(var)
        return series[idx] if series and idx < len(series) else None


def _points_key(prefix: str, points: Sequence[Tuple[float, float]], *parts: str) -> str:
    cells = ",".join(geohash(lat, lon, 6) for lat, lon in points)
    return f"{prefix}:{hashlib.sha1(cells.encode()).hexdigest()[:16]}:" + ":".join(parts)


def split_response(data: Any, points: Sequence[Tuple[float, float]], source: str, url: str,
                   retrieved_at: str) -> List[PointSeries]:
    items = data if isinstance(data, list) else [data]
    if len(items) != len(points):
        raise UpstreamError(source, f"expected {len(points)} locations, got {len(items)}")
    out = []
    for (lat, lon), item in zip(points, items):
        hourly = dict(item.get("hourly", {}))
        times = hourly.pop("time", [])
        daily = dict(item.get("daily", {}))
        out.append(PointSeries(lat=lat, lon=lon, grid_lat=item.get("latitude", lat), grid_lon=item.get("longitude", lon),
                               times=times, hourly=hourly, daily=daily, timezone=item.get("timezone", "UTC"),
                               utc_offset_seconds=item.get("utc_offset_seconds", 0), source=source, url=url,
                               retrieved_at=retrieved_at))
    return out


class WeatherService:
    SOURCE = "Open-Meteo Forecast API"

    def __init__(self, http: HttpGateway, cache: Cache):
        self.settings = get_settings()
        self.http = http
        self.cache = cache

    async def hourly(self, points: Sequence[Tuple[float, float]], start: date, end: date) -> List[PointSeries]:
        if not points:
            return []
        key = _points_key("wx", points, start.isoformat(), end.isoformat())
        cached = await self.cache.get(key)
        if cached is not None:
            return cached
        from datetime import datetime, timezone
        url = f"{self.settings.WEATHER_API_URL}/forecast"
        params = {
            "latitude": ",".join(f"{p[0]:.4f}" for p in points),
            "longitude": ",".join(f"{p[1]:.4f}" for p in points),
            "hourly": ",".join(HOURLY_VARS),
            "daily": ",".join(DAILY_VARS),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timezone": "auto",
            "wind_speed_unit": "kmh",
        }
        data = await self.http.request("GET", url, source=self.SOURCE, params=params)
        series = split_response(data, points, self.SOURCE, HttpGateway.url_with_params(url, params),
                                datetime.now(timezone.utc).isoformat(timespec="seconds"))
        await self.cache.set(key, series, ttl=self.settings.TTL_WEATHER)
        return series
