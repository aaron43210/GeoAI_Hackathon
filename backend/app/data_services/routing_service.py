"""OSRM travel times: one /table call from the origin to all venues; /route geometry for the chosen one."""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import get_settings
from app.data_services.cache import Cache, geohash
from app.data_services.http import HttpGateway, UpstreamError

logger = logging.getLogger(__name__)


class RoutingService:
    SOURCE = "OSRM (Project OSRM demo server, free-flow, no live traffic)"

    def __init__(self, http: HttpGateway, cache: Cache):
        self.settings = get_settings()
        self.http = http
        self.cache = cache

    async def table(self, origin: Tuple[float, float], dests: Sequence[Tuple[float, float]],
                    profile: str = "driving") -> List[Optional[Dict[str, Any]]]:
        """Per destination: {minutes, km, url} or None when OSRM has no route. Raises if OSRM is down."""
        if not dests:
            return []
        cells = geohash(*origin, 7) + ">" + ",".join(geohash(*d, 7) for d in dests)
        key = f"osrm:{profile}:{hashlib.sha1(cells.encode()).hexdigest()[:16]}"
        cached = await self.cache.get(key)
        if cached is not None:
            return cached
        coords = ";".join(f"{lon:.5f},{lat:.5f}" for lat, lon in [origin, *dests])
        url = f"{self.settings.OSRM_URL}/table/v1/{profile}/{coords}"
        params = {"sources": "0", "annotations": "duration,distance"}
        data = await self.http.request("GET", url, source="OSRM", params=params)
        if data.get("code") != "Ok":
            raise UpstreamError("OSRM", data.get("message", data.get("code", "error")))
        durations = (data.get("durations") or [[]])[0]
        distances = (data.get("distances") or [[]])[0]
        full_url = HttpGateway.url_with_params(url, params)
        out: List[Optional[Dict[str, Any]]] = []
        for i in range(1, len(dests) + 1):
            dur = durations[i] if i < len(durations) else None
            dist = distances[i] if i < len(distances) else None
            out.append(None if dur is None else {"minutes": round(dur / 60.0, 1), "km": round((dist or 0) / 1000, 2),
                                                 "url": full_url, "source": self.SOURCE})
        await self.cache.set(key, out, ttl=self.settings.TTL_ROUTE)
        return out

    async def route_geometry(self, origin: Tuple[float, float], dest: Tuple[float, float],
                             profile: str = "driving") -> Optional[Dict[str, Any]]:
        key = f"osrm-route:{profile}:{geohash(*origin, 7)}:{geohash(*dest, 7)}"
        cached = await self.cache.get(key)
        if cached is not None:
            return cached
        url = f"{self.settings.OSRM_URL}/route/v1/{profile}/{origin[1]:.5f},{origin[0]:.5f};{dest[1]:.5f},{dest[0]:.5f}"
        try:
            data = await self.http.request("GET", url, source="OSRM",
                                           params={"overview": "simplified", "geometries": "geojson"})
        except UpstreamError:
            return None
        routes = data.get("routes") or []
        geom = routes[0]["geometry"] if routes else None
        if geom:
            await self.cache.set(key, geom, ttl=self.settings.TTL_ROUTE)
        return geom
