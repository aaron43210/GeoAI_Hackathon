"""Venue × hour scoring grid: the core place-and-time reasoning.

For every candidate venue and every hour inside the resolved time windows we build a cell from live values
(weather, air quality, wind-aware exposure, travel at that hour), run the three-valued safety rules, and compute
a weighted 0–100 score from sub-score curves defined in knowledge/scoring.yaml. Consecutive cells are grouped into
activity-length windows, which are ranked to pick the best (venue, time), alternatives, avoid ranges, and a
contrast ("why not at the other time").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from app.agent.time_resolver import Window
from app.data_services.weather_service import PointSeries
from app.geospatial.context import VenueFeatures
from app.geospatial.exposure import ExposureResult, exposure, traffic_exposure_factor, travel_multiplier
from app.knowledge import ActivityProfile, get_knowledge
from app.scoring.rules_engine import RuleHit, Tri, evaluate_rules

FACTOR_LABELS = {"thermal": "thermal comfort", "rain": "rain", "wind": "wind", "uv": "UV", "air": "air quality",
                 "exposure": "traffic/industry air (wind-aware)", "setting": "green/water setting",
                 "access": "travel time", "light": "daylight/light"}


@dataclass
class VenueCtx:
    id: str
    name: str
    lat: float
    lon: float
    kind: str
    tags: Dict[str, str]
    osm_url: str
    features: VenueFeatures
    wx: PointSeries
    aq: Optional[PointSeries] = None
    travel: Optional[Dict[str, Any]] = None
    straight_km: float = 0.0


@dataclass
class Cell:
    venue: int
    ts: str
    dt: datetime
    vars: Dict[str, Any]
    factors: Dict[str, Optional[float]]
    exposure: Optional[ExposureResult]
    hits: List[RuleHit]
    travel_min: Optional[float]
    score: float
    status: str                      # ok | caution | avoid
    raw_score: float = 0.0           # before the severity cap; breaks ties between capped windows


@dataclass
class WindowPick:
    venue: int
    cells: List[Cell]
    score: float
    status: str
    label: str
    concerns: List[RuleHit] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)

    @property
    def start(self) -> datetime:
        return self.cells[0].dt

    @property
    def end(self) -> datetime:
        return self.cells[-1].dt + timedelta(hours=1)


@dataclass
class GridResult:
    venues: List[VenueCtx]
    cells: Dict[int, List[Cell]]
    windows: List[WindowPick]
    best: Optional[WindowPick]
    alternatives: List[WindowPick]
    alt_time: Optional[WindowPick]
    least_bad: Optional[WindowPick]
    compare: List[Dict[str, Any]]
    avoid_ranges: List[Dict[str, Any]]
    contrast: Optional[Dict[str, Any]]
    limits: Dict[str, Any]
    weights: Dict[str, float]


# ---------------------------------------------------------------------- sub-scores
def _lin(x: float, x0: float, x1: float) -> float:
    """1 at x0, 0 at x1 (either direction), clipped."""
    if x0 == x1:
        return 1.0 if x <= x0 else 0.0
    return max(0.0, min(1.0, (x1 - x) / (x1 - x0)))


def sub_scores(act: ActivityProfile, v: Dict[str, Any], feats: VenueFeatures, expo: Optional[ExposureResult],
               travel_min: Optional[float], max_travel: float, limits: Dict[str, Any], sun: Optional[tuple],
               hour: float) -> Dict[str, Optional[float]]:
    sc = get_knowledge().scoring
    out: Dict[str, Optional[float]] = {}

    at = v.get("apparent_temperature")
    lo, hi = act.comfort["apparent_lo"], act.comfort["apparent_hi"]
    if at is None:
        out["thermal"] = None
    elif at < lo:
        out["thermal"] = _lin(lo - at, 0, sc["thermal"]["zero_below_lo_by"])
    elif at > hi:
        out["thermal"] = _lin(at - hi, 0, sc["thermal"]["zero_above_hi_by"])
    else:
        out["thermal"] = 1.0

    p, mm = v.get("precipitation_probability"), v.get("precipitation")
    out["rain"] = None if p is None else (1 - p / 100) * (_lin(mm or 0, 0, sc["rain"]["mm_full_penalty"]) if mm else 1.0)

    g = v.get("wind_gusts_10m")
    out["wind"] = None if g is None else _lin(g, act.wind_comfort_kmh, limits.get("gust_max", act.wind_comfort_kmh * 2))

    uv = v.get("uv_index")
    if uv is None:
        out["uv"] = None
    else:
        u = sc["uv"]
        eff = uv * (1 - u["shade_mitigation"] * feats.green_fraction)
        out["uv"] = u["floor"] + (1 - u["floor"]) * _lin(eff, u["full_until"], u["floor_at"])

    aqi = v.get("us_aqi")
    if aqi is None:
        out["air"] = None
    else:
        a = sc["air"]
        out["air"] = _lin(aqi, a["full_until"], a["zero_at"]) ** a["exertion_exponent"][act.exertion]

    out["exposure"] = None if expo is None else 1 - expo.index

    if not feats.available:
        out["setting"] = None
        out["exposure"] = None
    st = sc["setting"]
    water = 0.0 if feats.nearest_water_m is None else _lin(feats.nearest_water_m, st["water_full_m"],
                                                          get_knowledge().geo["water_search_m"])
    quiet = _lin(feats.major_road_len_300m, 0, st["quiet_road_len_ref_m"])
    s = act.setting
    if feats.available:
        out["setting"] = (s.get("green", 0) * feats.green_fraction + s.get("water", 0) * water + s.get("quiet", 0) * quiet) / \
            max(1e-9, sum(s.values()))

    if travel_min is None:
        out["access"] = None
    else:
        acc = sc["access"]
        out["access"] = _lin(travel_min / max(1.0, max_travel), acc["full_until_ratio"], acc["zero_at_ratio"])

    li = sc["light"]
    is_day = v.get("is_day")
    if act.golden_hour and sun:
        sunrise, sunset = sun
        golden = is_golden(hour, sunrise, sunset, li["golden_window_h"])
        light = li["golden"] if golden else (li["golden_day"] if is_day else li["night"])
        cc = v.get("cloud_cover")
        if golden and cc is not None and not (li["good_cloud_min"] <= cc <= li["good_cloud_max"]):
            light *= li["bad_cloud_factor"]
        out["light"] = light
    elif is_day is None:
        out["light"] = None
    else:
        out["light"] = li["day"] if is_day else (li["night_lit"] if feats.lit == "yes" else li["night"])
    return out


def is_golden(hour: float, sunrise: float, sunset: float, window_h: float) -> bool:
    """The hour slot [hour, hour+1) overlaps the golden window after sunrise or before sunset."""
    return (hour < sunrise + window_h and hour + 1 > sunrise) or (hour < sunset and hour + 1 > sunset - window_h)


# ---------------------------------------------------------------------- grid
def _sun_hours(wx: PointSeries) -> Dict[date, tuple]:
    out = {}
    for d, sr, ss in zip(wx.daily.get("time", []), wx.daily.get("sunrise", []), wx.daily.get("sunset", [])):
        if sr and ss:
            a, b = datetime.fromisoformat(sr), datetime.fromisoformat(ss)
            out[date.fromisoformat(d)] = (a.hour + a.minute / 60, b.hour + b.minute / 60)
    return out


def _cell_vars(wx: PointSeries, aq: Optional[PointSeries], i: int, daylight_required: bool) -> Dict[str, Any]:
    v = {k: (vals[i] if i < len(vals) else None) for k, vals in wx.hourly.items()}
    if aq is not None and wx.times[i] in aq.times:
        j = aq.times.index(wx.times[i])
        for k, vals in aq.hourly.items():
            v[k] = vals[j] if j < len(vals) else None
    sc = get_knowledge().scoring
    window = sc["recent_rain_window_h"]
    prev = [x for x in wx.hourly.get("precipitation", [])[max(0, i - window):i] if x is not None]
    v["rain_prev_12h_mm"] = round(sum(prev), 1) if len(prev) >= sc["recent_rain_min_hours"] else None
    v["daylight_required"] = 1 if daylight_required else 0
    return v


def build_grid(venues: List[VenueCtx], windows: List[Window], act: ActivityProfile, profile_ids: List[str],
               preferences: List[str], duration_min: int, max_travel_min: float, now: datetime) -> GridResult:
    kb = get_knowledge()
    sc = kb.scoring
    limits = kb.effective_limits(act.id, profile_ids)
    weights = kb.effective_weights(act.id, profile_ids)
    pref_map = {"quiet": "setting", "shade": "uv", "water": "setting", "avoid_traffic": "exposure", "close_by": "access",
                "scenic": "setting"}
    for p in preferences:
        if pref_map.get(p) in weights:
            weights[pref_map[p]] *= sc["preference_boost"]
    daylight_required = act.daylight == "required" or any(kb.profiles[p].force_daylight for p in profile_ids
                                                          if p in kb.profiles)
    max_travel = max_travel_min * act.travel_tolerance * min(
        [kb.profiles[p].max_travel_factor for p in profile_ids if p in kb.profiles] or [1.0])

    cells: Dict[int, List[Cell]] = {}
    for vi, ven in enumerate(venues):
        sun = _sun_hours(ven.wx)
        wanted = set()
        for w in windows:
            first = math.ceil(w.start_h - kb.time["rules"]["min_lead_h"])
            h = first
            while h < w.end_h:
                wanted.add(datetime.combine(w.day, datetime.min.time()) + timedelta(hours=h))
                h += 1
        row: List[Cell] = []
        for i, ts in enumerate(ven.wx.times):
            dt = datetime.fromisoformat(ts)
            if dt not in wanted or dt < now.replace(minute=0, second=0, microsecond=0):
                continue
            v = _cell_vars(ven.wx, ven.aq, i, daylight_required)
            hour = dt.hour + dt.minute / 60
            expo = exposure(ven.features.sources, v.get("wind_direction_10m"), v.get("wind_speed_10m"),
                            traffic_exposure_factor(hour, dt.weekday())) if ven.features.available else None
            if expo is not None:
                v["exposure_index"] = expo.index
            travel_min = None
            if ven.travel:
                travel_min = round(ven.travel["minutes"] * travel_multiplier(hour, dt.weekday()))
            factors = sub_scores(act, v, ven.features, expo, travel_min, max_travel, limits, sun.get(dt.date()), hour)
            hits = evaluate_rules(kb.rules, v, limits)
            avail = {k: weights.get(k, 0) for k, s in factors.items() if s is not None and weights.get(k, 0) > 0}
            total_w = sum(avail.values())
            base = 100 * sum(factors[k] * w for k, w in avail.items()) / total_w if total_w else 0.0
            penalty = sum(h.penalty for h in hits if h.decision == "caution" and h.state == Tri.TRUE)
            status = "avoid" if any(h.decision == "avoid" and h.state == Tri.TRUE for h in hits) else (
                "caution" if any(h.decision == "caution" and h.state == Tri.TRUE for h in hits) else "ok")
            row.append(Cell(venue=vi, ts=ts, dt=dt, vars=v, factors=factors, exposure=expo, hits=hits,
                            travel_min=travel_min, score=_capped(base - penalty, hits), status=status,
                            raw_score=round(max(0.0, min(100.0, base - penalty)), 1)))
        cells[vi] = row

    # group into activity-length windows inside each resolved window
    n = max(1, math.ceil(duration_min / 60))
    picks: List[WindowPick] = []
    lead = kb.time["rules"]["min_lead_h"]
    for vi, row in cells.items():
        for w in windows:
            seg = [c for c in row if c.dt.date() == w.day and math.ceil(w.start_h - lead) <= c.dt.hour < w.end_h]
            if not seg:
                continue
            size = min(n, len(seg))
            for s in range(0, len(seg) - size + 1):
                group = seg[s:s + size]
                if any((group[k + 1].dt - group[k].dt) != timedelta(hours=1) for k in range(len(group) - 1)):
                    continue
                concerns = [h for c in group for h in c.hits if h.state == Tri.TRUE and h.decision == "caution"]
                unknown = sorted({h.var for c in group for h in c.hits if h.state == Tri.UNKNOWN and h.var})
                status = "avoid" if any(c.status == "avoid" for c in group) else (
                    "caution" if concerns else "ok")
                picks.append(WindowPick(venue=vi, cells=group, score=round(sum(c.score for c in group) / len(group), 1),
                                        status=status, label=w.label, concerns=_dedupe(concerns), unknowns=unknown))

    def rank_key(p: WindowPick):
        t = venues[p.venue].travel["minutes"] if venues[p.venue].travel else float("inf")
        raw = sum(c.raw_score for c in p.cells) / len(p.cells)
        return (-p.score, -raw, t, p.start)

    eligible = sorted([p for p in picks if p.status != "avoid"], key=rank_key)
    best = eligible[0] if eligible else None
    least_bad = None if best else (sorted(picks, key=rank_key)[0] if picks else None)

    alternatives: List[WindowPick] = []
    seen = {best.venue} if best else set()
    for p in eligible:
        if p.venue not in seen:
            alternatives.append(p)
            seen.add(p.venue)
        if len(alternatives) >= sc["alternatives"]:
            break

    alt_time = None
    if best:
        same = [p for p in eligible if p.venue == best.venue and (p.end <= best.start or p.start >= best.end)]
        other_day = [p for p in same if p.start.date() != best.start.date()]
        alt_time = (other_day or same or [None])[0]

    compare = []
    for d in sorted({w.day for w in windows}):
        day_picks = [p for p in picks if p.start.date() == d]
        ok = sorted([p for p in day_picks if p.status != "avoid"], key=rank_key)
        compare.append({"day": d, "best": ok[0] if ok else None, "gated_hours": sum(
            1 for p in day_picks if p.status == "avoid"), "windows": len(day_picks)})

    focus = best or least_bad
    avoid_ranges = _avoid_ranges(cells.get(focus.venue, [])) if focus else []
    contrast = _contrast(best, [p for p in picks if best and p.venue == best.venue], weights) if best else None
    return GridResult(venues=venues, cells=cells, windows=picks, best=best, alternatives=alternatives,
                      alt_time=alt_time, least_bad=least_bad, compare=compare, avoid_ranges=avoid_ranges,
                      contrast=contrast, limits=limits, weights=weights)


def _capped(score: float, hits: List[RuleHit]) -> float:
    caps = get_knowledge().scoring.get("severity_score_cap", {})
    for h in hits:
        if h.decision == "caution" and h.state == Tri.TRUE and h.severity in caps:
            score = min(score, caps[h.severity])
    return round(max(0.0, min(100.0, score)), 1)


def _dedupe(hits: List[RuleHit]) -> List[RuleHit]:
    """Keep the most severe value per rule id (the worst hour)."""
    by_id: Dict[str, RuleHit] = {}
    for h in hits:
        cur = by_id.get(h.id)
        if cur is None or (isinstance(h.value, (int, float)) and isinstance(cur.value, (int, float)) and
                           abs(h.value - (h.limit or 0)) > abs(cur.value - (cur.limit or 0))):
            by_id[h.id] = h
    return list(by_id.values())


def _avoid_ranges(row: List[Cell]) -> List[Dict[str, Any]]:
    ranges: List[Dict[str, Any]] = []
    for c in row:
        if c.status != "avoid":
            continue
        reasons = [h for h in c.hits if h.decision == "avoid" and h.state == Tri.TRUE]
        if ranges and ranges[-1]["end"] == c.dt and ranges[-1]["day"] == c.dt.date():
            ranges[-1]["end"] = c.dt + timedelta(hours=1)
            ranges[-1]["cells"].append(c)
            for r in reasons:
                if r.id not in {x.id for x in ranges[-1]["reasons"]}:
                    ranges[-1]["reasons"].append(r)
        else:
            ranges.append({"day": c.dt.date(), "start": c.dt, "end": c.dt + timedelta(hours=1), "reasons": reasons,
                           "cells": [c]})
    return ranges


def _contrast(best: WindowPick, same_venue: List[WindowPick], weights: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """The worst other window at the best venue, and the factor that explains most of the difference."""
    others = [p for p in same_venue if p.end <= best.start or p.start >= best.end]
    if not others:
        return None
    worst = min(others, key=lambda p: p.score)
    if best.score - worst.score < get_knowledge().scoring["contrast_min_gap"]:
        return None

    def mean(p: WindowPick, f: str) -> Optional[float]:
        vals = [c.factors.get(f) for c in p.cells if c.factors.get(f) is not None]
        return sum(vals) / len(vals) if vals else None

    drops = []
    for f, w in weights.items():
        a, b = mean(best, f), mean(worst, f)
        if a is not None and b is not None:
            drops.append(((a - b) * w, f))
    if not drops:
        return None
    _, factor = max(drops)
    return {"window": worst, "factor": factor, "label": FACTOR_LABELS.get(factor, factor),
            "gated": worst.status == "avoid"}
