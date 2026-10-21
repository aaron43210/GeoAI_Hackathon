"""Static geospatial features per venue, computed with shapely in a local metric projection.

All values derive from the OSM geometries fetched for this request; if a layer returned nothing, the feature
is reported as "none found within X m" (a real observation about OSM), never a guessed baseline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from app.data_services.osm_service import OSMBundle, OSMFeature
from app.knowledge import get_knowledge


class LocalProjection:
    """Equirectangular projection around a reference point; < 0.5 % distance error within ~50 km."""

    R = 6371000.0

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = math.radians(1) * self.R * math.cos(math.radians(lat0))
        self.ky = math.radians(1) * self.R

    def xy(self, lat: float, lon: float) -> Tuple[float, float]:
        return ((lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky)

    def latlon(self, x: float, y: float) -> Tuple[float, float]:
        return (self.lat0 + y / self.ky, self.lon0 + x / self.kx)


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def compass(deg: float) -> str:
    return COMPASS[int(((deg % 360) + 22.5) // 45) % 8]


def to_shape(f: OSMFeature, proj: LocalProjection):
    if f.geometry and len(f.geometry) >= 2:
        pts = [proj.xy(lat, lon) for lat, lon in f.geometry]
        closed = len(pts) >= 4 and pts[0] == pts[-1]
        if closed:
            poly = Polygon(pts)
            return poly if poly.is_valid else poly.buffer(0)
        return LineString(pts)
    return Point(proj.xy(f.lat, f.lon))


@dataclass
class SourcePoint:
    """A sampled emission source for the exposure model."""
    x: float
    y: float
    weight: float
    kind: str        # road | industry
    label: str


@dataclass
class VenueFeatures:
    area_m2: Optional[float]
    nearest_major_road_m: Optional[float]
    nearest_major_road_name: Optional[str]
    nearest_major_road_class: Optional[str]
    major_road_len_300m: float
    industry_within_3km: List[Dict[str, Any]]
    nearest_water_m: Optional[float]
    nearest_water_name: Optional[str]
    green_fraction: float
    power_line_within_m: Optional[float]
    lit: Optional[str]
    access: Optional[str]
    opening_hours: Optional[str]
    surface: Optional[str]
    sources: List[SourcePoint] = field(default_factory=list)
    available: bool = True          # False when OSM context could not be fetched (features are then not reported)

    @classmethod
    def unavailable(cls) -> "VenueFeatures":
        return cls(area_m2=None, nearest_major_road_m=None, nearest_major_road_name=None, nearest_major_road_class=None,
                   major_road_len_300m=0, industry_within_3km=[], nearest_water_m=None, nearest_water_name=None,
                   green_fraction=0.0, power_line_within_m=None, lit=None, access=None, opening_hours=None,
                   surface=None, available=False)


def area_m2(f: OSMFeature, proj: LocalProjection) -> Optional[float]:
    shape = to_shape(f, proj)
    if isinstance(shape, Polygon):
        return round(shape.area, 1)
    if f.bounds:   # relation: bounding-box area is an upper bound, marked as such by callers
        b = f.bounds
        x1, y1 = proj.xy(b["minlat"], b["minlon"])
        x2, y2 = proj.xy(b["maxlat"], b["maxlon"])
        return round(abs(x2 - x1) * abs(y2 - y1), 1)
    return None


def compute_features(venue: OSMFeature, ctx: OSMBundle) -> VenueFeatures:
    geo = get_knowledge().geo
    proj = LocalProjection(venue.lat, venue.lon)
    origin = Point(0, 0)
    venue_shape = to_shape(venue, proj)

    # Roads
    nearest_road = (None, None, None)
    major_len = 0.0
    buf300 = origin.buffer(300)
    sources: List[SourcePoint] = []
    for road in ctx.roads:
        shape = to_shape(road, proj)
        cls = road.tags.get("highway", "")
        weight = geo["road_classes"].get(cls, 0.0)
        d = shape.distance(origin)
        if nearest_road[0] is None or d < nearest_road[0]:
            nearest_road = (d, road.name or road.tags.get("ref"), cls)
        major_len += shape.intersection(buf300).length
        if isinstance(shape, LineString) and d < geo["context_radius_m"]:
            n = max(1, int(shape.length // 50))
            for i in range(n + 1):
                p = shape.interpolate(i / n, normalized=True)
                if p.distance(origin) < geo["context_radius_m"]:
                    sources.append(SourcePoint(p.x, p.y, weight / (n + 1) * max(1.0, shape.length / 500), "road",
                                               road.name or road.tags.get("ref") or cls))

    # Industry
    industry: List[Dict[str, Any]] = []
    for ind in ctx.industry:
        shape = to_shape(ind, proj)
        d = shape.distance(origin)
        if d > geo["context_radius_m"]:
            continue
        a = shape.area if isinstance(shape, Polygon) else 0.0
        c = shape.centroid
        lat, lon = proj.latlon(c.x, c.y)
        industry.append({"name": ind.name or ind.tags.get("landuse") or ind.tags.get("man_made") or "industrial site",
                         "distance_m": round(d), "bearing_deg": round(bearing_deg(venue.lat, venue.lon, lat, lon)),
                         "lat": round(lat, 5), "lon": round(lon, 5),
                         "area_m2": round(a), "osm_id": ind.osm_id})
        w = min(1.0, max(0.15, a / geo["industry_area_ref_m2"])) if a else 0.3
        sources.append(SourcePoint(c.x, c.y, w, "industry", industry[-1]["name"]))
    industry.sort(key=lambda r: r["distance_m"])

    # Water
    nearest_water = (None, None)
    for w in ctx.water:
        d = to_shape(w, proj).distance(venue_shape)
        if nearest_water[0] is None or d < nearest_water[0]:
            nearest_water = (d, w.name or w.tags.get("natural") or w.tags.get("waterway"))

    # Green cover: true polygon union ∩ buffer around the venue centroid
    buf = origin.buffer(geo["green_buffer_m"])
    greens = [to_shape(g, proj) for g in ctx.green]
    if isinstance(venue_shape, Polygon) and any(venue.tags.get(k) in v for k, v in geo["green_tags"].items()):
        greens.append(venue_shape)
    polys = [g for g in greens if isinstance(g, Polygon) and g.is_valid]
    green_frac = round(unary_union(polys).intersection(buf).area / buf.area, 3) if polys else 0.0

    # Power lines
    pl = [to_shape(p, proj).distance(venue_shape) for p in ctx.power_lines]

    return VenueFeatures(
        area_m2=area_m2(venue, proj),
        nearest_major_road_m=round(nearest_road[0]) if nearest_road[0] is not None else None,
        nearest_major_road_name=nearest_road[1],
        nearest_major_road_class=nearest_road[2],
        major_road_len_300m=round(major_len),
        industry_within_3km=industry[:5],
        nearest_water_m=round(nearest_water[0]) if nearest_water[0] is not None else None,
        nearest_water_name=nearest_water[1],
        green_fraction=green_frac,
        power_line_within_m=round(min(pl)) if pl else None,
        lit=venue.tags.get("lit"),
        access=venue.tags.get("access"),
        opening_hours=venue.tags.get("opening_hours"),
        surface=venue.tags.get("surface"),
        sources=sources,
    )
