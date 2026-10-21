"""OpenStreetMap via Overpass: candidate venues for an activity, then geospatial context around them.

Two queries per request (both cached by geohash):
  1. candidates: activity tag selectors within the search radius
  2. context: major roads, industry, water, green, power lines around the shortlisted venues
Mirrors are raced; if every mirror fails we raise UpstreamError. Nothing is ever synthesised.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.data_services.cache import Cache, geohash
from app.data_services.http import HttpGateway, UpstreamError
from app.knowledge import get_knowledge

logger = logging.getLogger(__name__)


@dataclass
class OSMFeature:
    osm_id: str
    tags: Dict[str, str]
    lat: float
    lon: float
    geometry: Optional[List[List[float]]] = None   # [[lat, lon], ...] for ways
    kind: str = "node"                              # node | way | relation
    bounds: Optional[Dict[str, float]] = None

    @property
    def name(self) -> Optional[str]:
        return self.tags.get("name:en") or self.tags.get("name")


@dataclass
class OSMBundle:
    candidates: List[OSMFeature] = field(default_factory=list)
    roads: List[OSMFeature] = field(default_factory=list)
    industry: List[OSMFeature] = field(default_factory=list)
    water: List[OSMFeature] = field(default_factory=list)
    green: List[OSMFeature] = field(default_factory=list)
    power_lines: List[OSMFeature] = field(default_factory=list)
    candidates_url: str = ""
    context_url: str = ""
    mirror: str = ""


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


class OSMService:
    def __init__(self, http: HttpGateway, cache: Cache):
        self.settings = get_settings()
        self.kb = get_knowledge()
        self.http = http
        self.cache = cache
        self._down_until = 0.0          # circuit breaker: skip Overpass for a while after every mirror failed

    # ------------------------------------------------------------------ queries
    async def _run(self, query: str) -> tuple[Dict[str, Any], str]:
        """Rolling race: keep N mirrors in flight; when one fails, start the next; first valid answer wins."""
        mirrors = list(self.settings.overpass_mirrors)
        if not mirrors:
            raise UpstreamError("Overpass", "no mirrors configured")
        if time.monotonic() < self._down_until:
            raise UpstreamError("Overpass", "skipped: all mirrors failed in the last few minutes")
        in_flight = max(1, int(self.kb.geo.get("overpass_rolling_race", 2)))

        async def one(url: str) -> tuple[Dict[str, Any], str]:
            data = await self.http.request("POST", url, source="Overpass", data={"data": query},
                                           timeout_s=self.settings.OVERPASS_TIMEOUT_S, retries=0)
            if not isinstance(data, dict) or "elements" not in data:
                raise UpstreamError("Overpass", f"unexpected response from {url}")
            remark = (data.get("remark") or "").lower()
            if "runtime error" in remark or "timed out" in remark:
                raise UpstreamError("Overpass", data["remark"][:160])
            return data, url

        def spawn(url: str) -> asyncio.Task:
            task = asyncio.create_task(one(url))
            task.add_done_callback(lambda t: t.cancelled() or t.exception())   # never leave errors unretrieved
            return task

        pending = {spawn(mirrors.pop(0)) for _ in range(min(in_flight, len(mirrors)))}
        errors: List[str] = []
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    result = task.result()
                except Exception as exc:  # noqa: BLE001 - try the next mirror
                    errors.append(str(exc))
                    if mirrors:
                        pending.add(spawn(mirrors.pop(0)))
                    continue
                for t in pending:
                    t.cancel()
                return result
        self._down_until = time.monotonic() + self.settings.OVERPASS_BREAKER_S
        raise UpstreamError("Overpass", "all mirrors failed: " + " | ".join(errors)[:400])

    async def find_candidates(self, activity: str, lat: float, lon: float, radius_km: float) -> OSMBundle:
        radius_m = int(radius_km * 1000)
        key = f"osm:cand:{activity}:{geohash(lat, lon, 6)}:{radius_m}"
        cached = await self.cache.get(key)
        if cached is not None:
            return cached

        selectors = self.kb.activities[activity].venues
        body = "\n".join(f"  {sel}(around:{radius_m},{lat:.5f},{lon:.5f});" for sel in selectors)
        query = f"[out:json][timeout:{int(self.settings.OVERPASS_TIMEOUT_S) - 5}];\n(\n{body}\n);\nout tags bb 400;"
        data, mirror = await self._run(query)
        bundle = OSMBundle(candidates=self._parse(data), candidates_url=f"{mirror} (Overpass QL: {len(selectors)} selectors, around {radius_m} m)", mirror=mirror)
        await self.cache.set(key, bundle, ttl=self.settings.TTL_OSM)
        return bundle

    async def fetch_context(self, venues: List[OSMFeature]) -> OSMBundle:
        geo = self.kb.geo
        if not venues:
            return OSMBundle()
        key = "osm:ctx:" + ",".join(sorted(geohash(v.lat, v.lon, 7) for v in venues))
        cached = await self.cache.get(key)
        if cached is not None:
            return cached

        ids = {"node": [], "way": [], "relation": []}
        for v in venues:
            kind, _, num = v.osm_id.partition("/")
            ids.setdefault(kind, []).append(num)
        sel = " ".join(f"{k}(id:{','.join(n)});" for k, n in ids.items() if n)
        roads = "|".join(geo["road_classes"].keys())
        r_ind, r_road = geo["context_radius_m"], geo["road_context_m"]
        r_water, r_green = geo["water_search_m"], geo["green_buffer_m"] + geo["green_query_margin_m"]
        stmts = [f'way["highway"~"^({roads})$"](around.c:{r_road});']
        for k, vals in geo["industry_tags"].items():
            stmts.append(f'nwr["{k}"~"^({"|".join(vals)})$"](around.c:{r_ind});')
        for k, vals in geo["water_tags"].items():
            stmts.append(f'way["{k}"~"^({"|".join(vals)})$"](around.c:{r_water});')
        for k, vals in geo["green_tags"].items():
            stmts.append(f'way["{k}"~"^({"|".join(vals)})$"](around.c:{r_green});')
        stmts.append(f'way["power"="line"](around.c:{geo["power_line_m"]});')
        query = (f"[out:json][timeout:{int(self.settings.OVERPASS_TIMEOUT_S) - 5}][maxsize:134217728];\n"
                 f"({sel})->.c;\n(\n  " + "\n  ".join(stmts) + "\n);\nout tags geom;")
        data, mirror = await self._run(query)
        feats = self._parse(data)
        bundle = OSMBundle(context_url=f"{mirror} (Overpass QL: roads/industry/water/green/power around {len(venues)} venues)", mirror=mirror)
        for f in feats:
            t = f.tags
            if t.get("highway") in geo["road_classes"]:
                bundle.roads.append(f)
            elif t.get("power") == "line":
                bundle.power_lines.append(f)
            elif any(t.get(k) in vals for k, vals in geo["industry_tags"].items()):
                bundle.industry.append(f)
            elif any(t.get(k) in vals for k, vals in geo["water_tags"].items()):
                bundle.water.append(f)
            elif any(t.get(k) in vals for k, vals in geo["green_tags"].items()):
                bundle.green.append(f)
        await self.cache.set(key, bundle, ttl=self.settings.TTL_OSM)
        return bundle

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def _parse(data: Dict[str, Any]) -> List[OSMFeature]:
        out: List[OSMFeature] = []
        for el in data.get("elements", []):
            etype = el.get("type")
            tags = el.get("tags", {})
            geom = [[p["lat"], p["lon"]] for p in el.get("geometry", []) if p]
            if etype == "node":
                lat, lon = el.get("lat"), el.get("lon")
            else:
                c = el.get("center") or {}
                lat, lon = c.get("lat"), c.get("lon")
                b = el.get("bounds")
                if lat is None and geom:
                    lat = sum(p[0] for p in geom) / len(geom)
                    lon = sum(p[1] for p in geom) / len(geom)
                elif lat is None and b:
                    lat = (b["minlat"] + b["maxlat"]) / 2
                    lon = (b["minlon"] + b["maxlon"]) / 2
            if lat is None or lon is None:
                continue
            out.append(OSMFeature(osm_id=f"{etype}/{el.get('id')}", tags=tags, lat=float(lat), lon=float(lon),
                                  geometry=geom or None, kind=etype, bounds=el.get("bounds")))
        return out
