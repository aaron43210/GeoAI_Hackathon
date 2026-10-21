"""Open-Meteo Air Quality (CAMS) hourly forecast for many points, plus optional WAQI station observation."""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import get_settings
from app.data_services.cache import Cache, geohash
from app.data_services.http import HttpGateway, UpstreamError
from app.data_services.weather_service import PointSeries, _points_key, split_response

logger = logging.getLogger(__name__)

AQ_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "us_aqi", "us_aqi_pm2_5", "us_aqi_pm10",
           "us_aqi_ozone", "us_aqi_nitrogen_dioxide"]


class AirQualityService:
    SOURCE = "Open-Meteo Air Quality API (CAMS)"

    def __init__(self, http: HttpGateway, cache: Cache):
        self.settings = get_settings()
        self.http = http
        self.cache = cache

    async def hourly(self, points: Sequence[Tuple[float, float]], start: date, end: date) -> List[PointSeries]:
        if not points:
            return []
        key = _points_key("aq", points, start.isoformat(), end.isoformat())
        cached = await self.cache.get(key)
        if cached is not None:
            return cached
        url = f"{self.settings.AIR_QUALITY_API_URL}/air-quality"
        params = {
            "latitude": ",".join(f"{p[0]:.4f}" for p in points),
            "longitude": ",".join(f"{p[1]:.4f}" for p in points),
            "hourly": ",".join(AQ_VARS),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timezone": "auto",
        }
        data = await self.http.request("GET", url, source=self.SOURCE, params=params)
        series = split_response(data, points, self.SOURCE, HttpGateway.url_with_params(url, params),
                                datetime.now(timezone.utc).isoformat(timespec="seconds"))
        await self.cache.set(key, series, ttl=self.settings.TTL_AIR)
        return series


class WAQIService:
    """Nearest ground-station reading (CPCB and others) from the World Air Quality Index project. Optional."""
    SOURCE = "WAQI (World Air Quality Index Project)"

    def __init__(self, http: HttpGateway, cache: Cache):
        self.settings = get_settings()
        self.http = http
        self.cache = cache

    @property
    def enabled(self) -> bool:
        return bool(self.settings.WAQI_TOKEN)

    async def nearest(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        key = f"waqi:{geohash(lat, lon, 5)}"
        cached = await self.cache.get(key)
        if cached is not None:
            return cached
        url = f"{self.settings.WAQI_URL}/feed/geo:{lat:.4f};{lon:.4f}/"
        try:
            data = await self.http.request("GET", url, source=self.SOURCE, params={"token": self.settings.WAQI_TOKEN})
        except UpstreamError as exc:
            logger.warning("WAQI unavailable: %s", exc)
            return None
        if not isinstance(data, dict) or data.get("status") != "ok":
            return None
        d = data["data"]
        city = d.get("city", {})
        geo = city.get("geo") or [None, None]
        result = {
            "station": city.get("name"),
            "station_lat": geo[0],
            "station_lon": geo[1],
            "aqi": d.get("aqi") if isinstance(d.get("aqi"), (int, float)) else None,
            "pm25": (d.get("iaqi", {}).get("pm25") or {}).get("v"),
            "observed_at": (d.get("time") or {}).get("iso"),
            "url": url + "?token=***",
        }
        await self.cache.set(key, result, ttl=self.settings.TTL_WAQI)
        return result
