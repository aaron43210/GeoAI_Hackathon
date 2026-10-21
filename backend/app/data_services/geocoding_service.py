import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.data_services.cache import Cache
from app.data_services.http import HttpGateway, UpstreamError

logger = logging.getLogger(__name__)


@dataclass
class Place:
    name: str
    display_name: str
    latitude: float
    longitude: float
    bbox: Optional[List[float]] = None          # [south, north, west, east]
    city: Optional[str] = None
    state: Optional[str] = None
    osm_type: Optional[str] = None
    importance: float = 0.0
    source_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class GeocodeResult:
    query: str
    best: Optional[Place]
    candidates: List[Place] = field(default_factory=list)
    ambiguous: bool = False


class GeocodingService:
    """Nominatim geocoding. Never substitutes a default location: unresolved means unresolved."""

    def __init__(self, http: HttpGateway, cache: Cache):
        self.settings = get_settings()
        self.http = http
        self.cache = cache

    async def geocode(self, query: str) -> GeocodeResult:
        cleaned = " ".join(query.split()).strip()
        if not cleaned:
            return GeocodeResult(query=query, best=None)
        key = f"geo:{self.settings.GEOCODE_COUNTRY_CODES}:{cleaned.lower()}"
        cached = await self.cache.get(key)
        if cached is not None:
            return cached

        url = f"{self.settings.NOMINATIM_URL}/search"
        params = {"q": cleaned, "format": "jsonv2", "limit": 5, "addressdetails": 1}
        if self.settings.GEOCODE_COUNTRY_CODES:
            params["countrycodes"] = self.settings.GEOCODE_COUNTRY_CODES
        data = await self.http.request("GET", url, source="Nominatim", params=params)
        source_url = HttpGateway.url_with_params(url, params)

        places = [self._to_place(item, source_url) for item in data or []]
        places.sort(key=lambda p: (self._type_rank(p), -p.importance))
        result = GeocodeResult(query=cleaned, best=places[0] if places else None, candidates=places,
                               ambiguous=self._is_ambiguous(places))
        await self.cache.set(key, result, ttl=self.settings.TTL_GEOCODE if places else self.settings.TTL_NEGATIVE)
        return result

    async def search_category(self, phrase: str, lat: float, lon: float, radius_km: float) -> List[Dict[str, Any]]:
        """OSM features of a category (Nominatim special phrase) inside a bounding box: the venue fallback."""
        import math
        dlat = radius_km / 111.0
        dlon = radius_km / (111.0 * max(0.2, math.cos(math.radians(lat))))
        key = f"cat:{phrase}:{round(lat, 3)}:{round(lon, 3)}:{radius_km}"
        cached = await self.cache.get(key)
        if cached is not None:
            return cached
        url = f"{self.settings.NOMINATIM_URL}/search"
        params = {"q": phrase, "format": "jsonv2", "limit": 40, "bounded": 1, "extratags": 1,
                  "viewbox": f"{lon - dlon:.4f},{lat + dlat:.4f},{lon + dlon:.4f},{lat - dlat:.4f}"}
        data = await self.http.request("GET", url, source="Nominatim", params=params)
        await self.cache.set(key, data or [], ttl=self.settings.TTL_OSM)
        return data or []

    async def reverse(self, lat: float, lon: float) -> Optional[Place]:
        key = f"rgeo:{round(lat, 4)}:{round(lon, 4)}"
        cached = await self.cache.get(key)
        if cached is not None:
            return cached
        url = f"{self.settings.NOMINATIM_URL}/reverse"
        params = {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 14, "addressdetails": 1}
        try:
            data = await self.http.request("GET", url, source="Nominatim", params=params)
        except UpstreamError:
            return None
        place = self._to_place(data, HttpGateway.url_with_params(url, params)) if data and "lat" in data else None
        if place:
            place.latitude, place.longitude = lat, lon
            await self.cache.set(key, place, ttl=self.settings.TTL_GEOCODE)
        return place

    @staticmethod
    def _to_place(item: Dict[str, Any], source_url: str) -> Place:
        addr = item.get("address", {})
        bbox = item.get("boundingbox")
        return Place(
            name=item.get("name") or item.get("display_name", "").split(",")[0],
            display_name=item.get("display_name", ""),
            latitude=float(item["lat"]),
            longitude=float(item["lon"]),
            bbox=[float(b) for b in bbox] if bbox else None,
            city=addr.get("city") or addr.get("town") or addr.get("village") or addr.get("county")
                 or addr.get("state_district"),
            state=addr.get("state"),
            osm_type=item.get("addresstype") or item.get("type"),
            importance=float(item.get("importance") or 0.0),
            source_url=source_url,
        )

    @staticmethod
    def _type_rank(p: Place) -> int:
        from app.knowledge import get_knowledge
        order = get_knowledge().geo["place_type_priority"]
        return order.index(p.osm_type) if p.osm_type in order else len(order)

    @classmethod
    def _is_ambiguous(cls, places: List[Place]) -> bool:
        """Ambiguous only when the top two hits are the same kind of place, in different states, equally important."""
        if len(places) < 2:
            return False
        a, b = places[0], places[1]
        if not a.state or not b.state or a.state == b.state or cls._type_rank(a) != cls._type_rank(b):
            return False
        from app.knowledge import get_knowledge
        return b.importance >= get_knowledge().geo["place_ambiguity_ratio"] * a.importance
