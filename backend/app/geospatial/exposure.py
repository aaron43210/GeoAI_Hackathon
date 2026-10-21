"""Wind-aware exposure model: how much road/industrial air is blown towards a venue at a given hour.

For each sampled source s at distance d and bearing β from the venue, with wind blowing FROM θ:
    alignment = cos²(θ − β)  if |θ − β| < cone, else 0     (source is upwind of the venue)
    contribution = alignment · weight · e^(−d / decay) · traffic_factor(h)   (traffic factor for roads only)
In calm wind (< calm_wind_kmh) dispersion is weak, so every source counts at half weight regardless of direction.
The index is clipped to [0, 1]. It is a MODELLED relative index, not a measurement, and is labelled so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from app.geospatial.context import SourcePoint, compass
from app.knowledge import get_knowledge

FORMULA = ("exposure = Σ cos²(wind_from − bearing_to_source)·weight·exp(−d/decay)·traffic(h), "
           "sources within ±cone° upwind; calm wind → half weight all directions; clipped to [0,1]")


@dataclass
class ExposureResult:
    index: float
    road_part: float
    industry_part: float
    dominant_source: Optional[str]
    dominant_kind: Optional[str]
    upwind_compass: Optional[str]


def exposure(sources: List[SourcePoint], wind_from_deg: Optional[float], wind_kmh: Optional[float],
             traffic_factor: float) -> Optional[ExposureResult]:
    if wind_from_deg is None or wind_kmh is None:
        return None
    geo = get_knowledge().geo
    cone = geo["wind_cone_deg"]
    calm = wind_kmh < geo["calm_wind_kmh"]
    road = ind = 0.0
    best = (0.0, None, None)
    for s in sources:
        d = math.hypot(s.x, s.y)
        if d < 1:
            d = 1.0
        beta = (math.degrees(math.atan2(s.x, s.y)) + 360) % 360
        diff = abs((wind_from_deg - beta + 180) % 360 - 180)
        if calm:
            align = 0.5
        elif diff < cone:
            align = math.cos(math.radians(diff)) ** 2
        else:
            continue
        decay = geo["road_decay_m"] if s.kind == "road" else geo["industry_decay_m"]
        c = align * s.weight * math.exp(-d / decay) * (traffic_factor if s.kind == "road" else 1.0)
        if s.kind == "road":
            road += c
        else:
            ind += c
        if c > best[0]:
            best = (c, s.label, s.kind)
    total = min(geo["exposure_cap"], road + ind)
    return ExposureResult(index=round(total, 3), road_part=round(road, 3), industry_part=round(ind, 3),
                          dominant_source=best[1], dominant_kind=best[2],
                          upwind_compass=compass(wind_from_deg))


def traffic_exposure_factor(hour: float, weekday: int) -> float:
    t = get_knowledge().geo["traffic"]
    if weekday < 5:
        for band in t["exposure_weekday"]:
            if band["start"] <= hour < band["end"]:
                return band["factor"]
    return t["exposure_default"] if weekday < 5 else t["exposure_weekend"]


def travel_multiplier(hour: float, weekday: int) -> float:
    t = get_knowledge().geo["traffic"]
    bands = t["weekday_multiplier"] if weekday < 5 else t["weekend_multiplier"]
    for band in bands:
        if band["start"] <= hour < band["end"]:
            return band["factor"]
    return t["default_factor"]
