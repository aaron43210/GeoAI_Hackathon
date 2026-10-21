"""Practical advice (what to carry / what to watch for) from knowledge/advice.yaml, evaluated on window aggregates."""
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.knowledge import ActivityProfile, get_knowledge
from app.scoring.grid import VenueCtx, WindowPick, _sun_hours
from app.scoring.rules_engine import Tri, evaluate_condition


def _max(cells, var):
    vals = [c.vars.get(var) for c in cells if c.vars.get(var) is not None]
    return max(vals) if vals else None


def _min(cells, var):
    vals = [c.vars.get(var) for c in cells if c.vars.get(var) is not None]
    return min(vals) if vals else None


def aggregates(pick: WindowPick, venue: VenueCtx, act: ActivityProfile, duration_min: int,
               all_cells_same_day: List) -> Dict[str, Any]:
    sc = get_knowledge().scoring
    cells = pick.cells
    agg: Dict[str, Any] = {
        "max_apparent": _max(cells, "apparent_temperature"),
        "min_apparent": _min(cells, "apparent_temperature"),
        "max_uv": _max(cells, "uv_index"),
        "max_precip_prob": _max(cells, "precipitation_probability"),
        "max_gust": _max(cells, "wind_gusts_10m"),
        "max_us_aqi": _max(cells, "us_aqi"),
        "max_pm2_5": _max(cells, "pm2_5"),
        "min_visibility": _min(cells, "visibility"),
        "rain_prev_12h_mm": cells[0].vars.get("rain_prev_12h_mm"),
        "duration_min": duration_min,
        "ends_after_hour": pick.end.hour + pick.end.minute / 60,
        "water_within_m": venue.features.nearest_water_m,
        "storm_in_day": 1 if any((c.vars.get("weather_code") or 0) in sc["storm_codes"] for c in all_cells_same_day) else 0,
    }
    expos = [c.exposure for c in cells if c.exposure is not None]
    if expos:
        top = max(expos, key=lambda e: e.index)
        agg["exposure_max"] = top.index
        agg["exposure_source"] = top.dominant_source or "the busiest nearby road"
    aq = [c.vars.get("us_aqi") for c in cells if c.vars.get("us_aqi") is not None]
    if len(aq) >= 2:
        agg["aqi_start"], agg["aqi_end"] = aq[0], aq[-1]
        agg["aqi_rising"] = 1 if aq[-1] - aq[0] >= sc["aqi_rising_delta"] else 0
    sun = _sun_hours(venue.wx).get(pick.start.date())
    if sun:
        agg["sunset"] = f"{int(sun[1]):02d}:{int(round((sun[1] % 1) * 60)) % 60:02d}"
        agg["minutes_to_sunset_after_end"] = round((sun[1] - agg["ends_after_hour"]) * 60)
    per30 = sc["water_litres_per_30min"][act.exertion]
    hot = (agg["max_apparent"] or 0) >= sc["hot_threshold_apparent"]
    litres = per30 * math.ceil(duration_min / 30) * (sc["hot_water_factor"] if hot else 1.0)
    agg["water_litres"] = round(litres * 2) / 2 if litres >= 1 else round(litres, 1)
    return agg


def _applies(item: Dict[str, Any], act_id: str, profiles: List[str]) -> bool:
    prof = item.get("profile")
    if prof is not None:
        wanted = prof if isinstance(prof, list) else [prof]
        if not set(wanted) & set(profiles):
            return False
    acts = item.get("activity")
    if acts is not None:
        wanted = acts if isinstance(acts, list) else [acts]
        if act_id not in wanted:
            return False
    return True


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return str(v)


def build_advice(agg: Dict[str, Any], act_id: str, profiles: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    kb = get_knowledge()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for section in ("carry", "watch_for"):
        items = []
        for item in kb.advice.get(section, []):
            if not _applies(item, act_id, profiles):
                continue
            if not item.get("always"):
                state, *_ = evaluate_condition(item["when"], agg, {})
                if state != Tri.TRUE:
                    continue
            used = re.findall(r"{(\w+)}", item["text"])
            if any(agg.get(u) is None for u in used):
                continue
            text = item["text"].format(**{u: _fmt(agg[u]) for u in used})
            items.append({"id": item["id"], "text": text, "vars": used})
        out[section] = items
    return out
