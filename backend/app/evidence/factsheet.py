"""Builds the evidence ledger and the compact FACT SHEET the SLM writes from (and the verifier checks against)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.advice.rules import aggregates, build_advice
from app.data_services.osm_service import OSMFeature
from app.evidence.ledger import Ledger
from app.geospatial.context import compass
from app.geospatial.exposure import FORMULA as EXPOSURE_FORMULA
from app.knowledge import ActivityProfile, get_knowledge
from app.scoring.grid import FACTOR_LABELS, GridResult, VenueCtx, WindowPick, _sun_hours


def fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return str(v)


def fmt_window(p: WindowPick) -> str:
    return f"{p.start:%a %d %b} {p.start:%H:%M}–{p.end:%H:%M}"


def osm_url(osm_id: str) -> str:
    return f"https://www.openstreetmap.org/{osm_id}"


@dataclass
class FactSheet:
    text: str
    ledger: Ledger
    advice: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    aggregates: Dict[str, Any] = field(default_factory=dict)
    allowed_names: List[str] = field(default_factory=list)
    avoid: List[Dict[str, Any]] = field(default_factory=list)


class FactSheetBuilder:
    def __init__(self, grid: GridResult, act: ActivityProfile, profiles: List[str], question: str,
                 place_label: str, duration_min: int, notes: List[str], limitations: List[str],
                 waqi: Optional[Dict[str, Any]] = None):
        self.g, self.act, self.profiles, self.q = grid, act, profiles, question
        self.place_label, self.duration, self.notes, self.limitations, self.waqi = \
            place_label, duration_min, notes, limitations, waqi
        self.L = Ledger()

    # -------------------------------------------------------------- facts
    def _venue(self, v: VenueCtx) -> str:
        return self.L.add(v.name, v.name, None, "OpenStreetMap (Overpass)", v.osm_url, "osm", key=f"venue:{v.id}",
                          venue=v.name, method=f"OSM {v.kind} tagged " + ", ".join(
                              f"{k}={v.tags[k]}" for k in ("leisure", "sport", "tourism", "natural", "historic") if k in v.tags))

    def _wx(self, v: VenueCtx, p: WindowPick, var: str, label: str, unit: str, agg: str = "range",
            src: str = "wx") -> Optional[str]:
        vals = [c.vars.get(var) for c in p.cells if c.vars.get(var) is not None]
        if not vals:
            return None
        if agg == "max":
            value, text = max(vals), fmt(max(vals))
        elif agg == "first":
            value, text = vals[0], fmt(vals[0])
        else:
            lo, hi = min(vals), max(vals)
            value, text = (lo, hi), fmt(lo) if lo == hi else f"{fmt(lo)}–{fmt(hi)}"
        series = v.wx if src == "wx" else v.aq
        eid = self.L.add(label, value, unit, series.source, series.url,
                         "forecast", key=f"{var}:{agg}:{v.id}:{p.start}", time=fmt_window(p), venue=v.name,
                         retrieved_at=series.retrieved_at,
                         method=f"hourly model value at grid point {series.grid_lat:.3f},{series.grid_lon:.3f}")
        return f"[{eid}] {label} {text}{(' ' + unit) if unit else ''} ({p.start:%H:%M}–{p.end:%H:%M})"

    def _pick_lines(self, p: WindowPick, primary: bool) -> List[str]:
        v = self.g.venues[p.venue]
        lines: List[str] = []
        add = lambda x: x and lines.append(x)  # noqa: E731
        add(self._wx(v, p, "apparent_temperature", "feels-like temperature", "°C"))
        add(self._wx(v, p, "precipitation_probability", "rain probability up to", "%", "max"))
        if v.aq is not None:
            line = self._wx(v, p, "us_aqi", "US AQI up to", "", "max", src="aq")
            if line:
                worst = max((c for c in p.cells if c.vars.get("us_aqi") is not None), key=lambda c: c.vars["us_aqi"])
                subs = {k: worst.vars.get(f"us_aqi_{k}") for k in ("pm2_5", "pm10", "ozone", "nitrogen_dioxide")}
                subs = {k: x for k, x in subs.items() if x is not None}
                if subs:
                    dom = max(subs, key=subs.get)
                    names = {"pm2_5": "PM2.5", "pm10": "PM10", "ozone": "ozone", "nitrogen_dioxide": "NO2"}
                    line += f"; main pollutant {names[dom]}" + (f", PM2.5 sub-index {subs['pm2_5']}" if dom != "pm2_5" and "pm2_5" in subs else "")
                    note = ("the air-quality index here is driven by the CAMS ozone forecast; ground-station "
                            "readings may differ")
                    if dom == "ozone" and note not in self.limitations:
                        self.limitations.insert(0, note)
            add(line)
            if primary:
                add(self._wx(v, p, "pm2_5", "PM2.5 up to", "µg/m³", "max", src="aq"))
        else:
            eid = self.L.add("air quality", "unavailable", None, "Open-Meteo Air Quality API (CAMS)", None, "observation",
                             key=f"aq-missing:{v.id}")
            lines.append(f"[{eid}] air quality forecast: unavailable (request failed)")
        if not primary:
            return lines
        c0 = p.cells[0]
        wd, ws = c0.vars.get("wind_direction_10m"), c0.vars.get("wind_speed_10m")
        if wd is not None and ws is not None:
            eid = self.L.add("wind", {"from_deg": wd, "kmh": ws}, "km/h", v.wx.source, v.wx.url, "forecast",
                             key=f"wind:{v.id}:{p.start}", time=f"{c0.dt:%H:%M}", venue=v.name)
            lines.append(f"[{eid}] wind from {compass(wd)} at {fmt(ws)} km/h ({c0.dt:%H:%M})")
        add(self._wx(v, p, "wind_gusts_10m", "gusts up to", "km/h", "max"))
        add(self._wx(v, p, "uv_index", "UV index up to", "", "max"))
        expos = [c.exposure for c in p.cells if c.exposure]
        if expos:
            e = max(expos, key=lambda x: x.index)
            src = f", main upwind source: {e.dominant_source}" \
                if e.dominant_source and e.index >= get_knowledge().geo["exposure_significant"] else \
                ", no significant upwind road/industry"
            eid = self.L.add("modelled exposure index", e.index, "0–1", "computed from OSM roads/industry + forecast wind",
                             None, "modelled", key=f"expo:{v.id}:{p.start}", venue=v.name, method=EXPOSURE_FORMULA)
            lines.append(f"[{eid}] traffic/industry exposure index {fmt(e.index)} (modelled, 0 = none){src}")
        f = v.features
        if not f.available:
            eid = self.L.add("map context", "unavailable", None, "OpenStreetMap (Overpass)", None, "osm",
                             key=f"ctx-missing:{v.id}")
            lines.append(f"[{eid}] roads/industry/green/water around the place: unavailable (map query failed)")
        if f.available and f.nearest_major_road_m is not None:
            eid = self.L.add("nearest major road", f.nearest_major_road_m, "m", "OpenStreetMap (Overpass)",
                             self.g.venues[p.venue].osm_url, "osm", key=f"road:{v.id}", venue=v.name,
                             method="shapely distance from venue centroid to OSM motorway/trunk/primary/secondary ways")
            lines.append(f"[{eid}] nearest major road: {f.nearest_major_road_name or f.nearest_major_road_class} "
                         f"{f.nearest_major_road_m} m away")
        if f.available:
            eid = self.L.add("green cover", round(f.green_fraction * 100), "%", "OpenStreetMap (Overpass)", None,
                             "computed", key=f"green:{v.id}", venue=v.name,
                             method="area of OSM park/forest/grass polygons ∩ 300 m buffer ÷ buffer area")
            lines.append(f"[{eid}] green cover within {get_knowledge().geo['green_buffer_m']} m: "
                         f"{round(f.green_fraction * 100)}%")
        if f.nearest_water_m is not None:
            eid = self.L.add("nearest water", f.nearest_water_m, "m", "OpenStreetMap (Overpass)", None, "osm",
                             key=f"water:{v.id}", venue=v.name)
            lines.append(f"[{eid}] water body {f.nearest_water_name or ''} {f.nearest_water_m} m away".replace("  ", " "))
        if f.industry_within_3km:
            ind = f.industry_within_3km[0]
            eid = self.L.add("nearest industry", ind["distance_m"], "m", "OpenStreetMap (Overpass)",
                             osm_url(ind["osm_id"]), "osm", key=f"ind:{v.id}", venue=v.name)
            lines.append(f"[{eid}] nearest industrial site {ind['name']} {ind['distance_m']} m to the "
                         f"{compass(ind['bearing_deg'])}")
        if v.travel:
            rush = p.cells[0].travel_min
            eid = self.L.add("travel time", v.travel["minutes"], "min", v.travel["source"], v.travel["url"], "computed",
                             key=f"travel:{v.id}", venue=v.name)
            eid2 = self.L.add("travel time with traffic", rush, "min", "heuristic rush-hour multiplier (geo.yaml)",
                              None, "heuristic", key=f"rush:{v.id}:{p.start}")
            lines.append(f"[{eid}] {fmt(v.travel['minutes'])} min / {fmt(v.travel['km'])} km by road (free-flow); "
                         f"[{eid2}] about {fmt(rush)} min at {p.start:%H:%M} with typical traffic (estimate)")
        elif v.kind != "area":
            eid = self.L.add("travel time", "unavailable", None, "OSRM", None, "computed", key=f"travel-missing:{v.id}")
            lines.append(f"[{eid}] travel time unavailable; straight-line distance {fmt(round(v.straight_km, 1))} km")
        sun = _sun_hours(v.wx).get(p.start.date())
        if sun:
            sr = f"{int(sun[0]):02d}:{int(round((sun[0] % 1) * 60)) % 60:02d}"
            ss = f"{int(sun[1]):02d}:{int(round((sun[1] % 1) * 60)) % 60:02d}"
            eid = self.L.add("sunrise/sunset", {"sunrise": sr, "sunset": ss}, None, v.wx.source, v.wx.url, "forecast",
                             key=f"sun:{v.id}:{p.start.date()}")
            lines.append(f"[{eid}] sunrise {sr}, sunset {ss} on {p.start:%a %d %b}")
        for h in p.concerns:
            eid = self.L.add(h.id, h.value, None, "rule " + h.id + " on forecast values", None, "derived",
                             key=f"concern:{h.id}:{v.id}:{p.start}")
            lines.append(f"[{eid}] caution: {h.text}")
        for u in p.unknowns:
            eid = self.L.add(f"{u} unavailable", None, None, "missing upstream value", None, "derived",
                             key=f"unk:{u}:{v.id}")
            lines.append(f"[{eid}] {u.replace('_', ' ')}: data unavailable, could not be checked")
        return lines

    # -------------------------------------------------------------- sheet
    def build(self) -> FactSheet:
        g = self.g
        kb = get_knowledge()
        lines: List[str] = [f"QUESTION: {self.q}"]
        prof_txt = ", ".join(kb.profiles[p].label for p in self.profiles) or "none stated"
        lim = g.limits
        lines.append(f"ACTIVITY: {self.act.label}; duration {self.duration} min; people: {prof_txt}")
        lines.append(f"LIMITS USED: feels-like ≤ {fmt(lim.get('apparent_max'))} °C, US AQI ≤ {fmt(lim.get('us_aqi_max'))}, "
                     f"gusts ≤ {fmt(lim.get('gust_max'))} km/h, rain probability ≤ {fmt(lim.get('precip_prob_max'))}%")
        lines.append(f"AREA: {self.place_label}; {len(g.venues)} candidate places checked across "
                     f"{len({c.ts for row in g.cells.values() for c in row})} hours")
        for n in self.notes:
            lines.append(f"NOTE: {n}")

        sheet = FactSheet(text="", ledger=self.L)
        focus = g.best or g.least_bad
        if focus is None:
            lines.append("RESULT: no forecast hours available for the requested time.")
            sheet.text = "\n".join(lines)
            return sheet
        v = g.venues[focus.venue]
        vid = self._venue(v)
        sid = self.L.add("suitability score", focus.score, "/100", "TerraPulse scoring grid", None, "computed",
                         key=f"score:{v.id}:{focus.start}", method="weighted sub-scores (knowledge/scoring.yaml)")
        if g.best:
            lines.append(f"BEST: [{vid}] {v.name}, {fmt_window(focus)}, score [{sid}] {fmt(focus.score)}/100")
            severe = [h for h in focus.concerns if h.severity == "high"]
            for h in severe:
                hid = self.L.add(h.id, h.value, None, "rule " + h.id + " on forecast values", None, "derived",
                                 key=f"warn:{h.id}:{v.id}:{focus.start}")
                lines.append(f"HEALTH WARNING: [{hid}] {h.text}")
        else:
            lines.append(f"RESULT: NOT RECOMMENDED in the requested time: every hour hits a safety limit. "
                         f"Least-bad option: [{vid}] {v.name}, {fmt_window(focus)} (score [{sid}] {fmt(focus.score)})")
        lines.append("FACTS:")
        lines += self._pick_lines(focus, primary=True)
        if self.waqi and self.waqi.get("aqi") is not None:
            eid = self.L.add("observed AQI", self.waqi["aqi"], "AQI", "WAQI ground station", self.waqi["url"],
                             "observation", time=self.waqi.get("observed_at"), venue=self.waqi.get("station"))
            lines.append(f"[{eid}] ground station {self.waqi['station']} observed AQI {self.waqi['aqi']} "
                         f"at {(self.waqi.get('observed_at') or '')[:16].replace('T', ' ')}")

        same_day = [c for c in g.cells.get(focus.venue, []) if c.dt.date() == focus.start.date()]
        agg = aggregates(focus, v, self.act, self.duration, same_day)
        advice = build_advice(agg, self.act.id, self.profiles)
        for sec, title in (("carry", "CARRY"), ("watch_for", "WATCH FOR")):
            if advice[sec]:
                lines.append(f"{title}: " + "; ".join(i["text"] for i in advice[sec]))

        avoid_lines = []
        for r in g.avoid_ranges[:4]:
            reasons = "; ".join(h.text for h in r["reasons"][:2])
            eid = self.L.add("avoid window", f"{r['start']:%H:%M}-{r['end']:%H:%M}", None,
                             "safety rules on forecast values", None, "derived", key=f"avoid:{r['start']}")
            avoid_lines.append(f"[{eid}] {r['start']:%a %H:%M}–{r['end']:%H:%M}: {reasons}")
            sheet.avoid.append({"start": r["start"].isoformat(), "end": r["end"].isoformat(),
                                "reasons": [h.text for h in r["reasons"]], "evidence": eid})
        if g.contrast:
            c = g.contrast
            w = c["window"]
            vals = [x.exposure.dominant_source for x in w.cells if x.exposure and x.exposure.dominant_source]
            extra = f" (wind brings air from {vals[0]})" if c["factor"] == "exposure" and vals else ""
            eid = self.L.add("contrast", w.score, "/100", "TerraPulse scoring grid", None, "computed",
                             key=f"contrast:{w.start}")
            avoid_lines.append(f"[{eid}] {fmt_window(w)} at the same place scores only {fmt(w.score)}/100, mainly "
                               f"because of {c['label']}{extra}")
        if avoid_lines:
            lines.append("AVOID / TIMING:")
            lines += avoid_lines

        others = []
        for alt in g.alternatives:
            av = g.venues[alt.venue]
            aid = self._venue(av)
            asid = self.L.add("suitability score", alt.score, "/100", "TerraPulse scoring grid", None, "computed",
                              key=f"score:{av.id}:{alt.start}")
            extra = self._pick_lines(alt, primary=False)
            dist = f", {fmt(av.travel['minutes'])} min away" if av.travel else ""
            others.append(f"[{aid}] {av.name}, {fmt_window(alt)}, score [{asid}] {fmt(alt.score)}{dist}; " +
                          "; ".join(extra))
        if g.alt_time:
            t = g.alt_time
            tid = self.L.add("suitability score", t.score, "/100", "TerraPulse scoring grid", None, "computed",
                             key=f"score:{v.id}:{t.start}")
            others.append(f"same place at {fmt_window(t)}: score [{tid}] {fmt(t.score)}")
        if len(g.compare) > 1:
            for d in g.compare:
                b = d["best"]
                if b:
                    did = self.L.add("day best score", b.score, "/100", "TerraPulse scoring grid", None, "computed",
                                     key=f"day:{d['day']}")
                    others.append(f"{d['day']:%A}: best {g.venues[b.venue].name} {b.start:%H:%M} score [{did}] {fmt(b.score)}")
                else:
                    others.append(f"{d['day']:%A}: no safe window ({d['gated_hours']} windows blocked)")
        if others:
            lines.append("OTHER OPTIONS:")
            lines += others
        oh = g.venues[focus.venue].features.opening_hours if focus else None
        avail = ("opening hours (OSM: " + oh + ") and bookings were not checked live") if oh else \
            "opening hours, entry rules and bookings (e.g. tee times, court slots) were not checked"
        if avail not in self.limitations:
            self.limitations.append(avail)
        if self.limitations:
            lines.append("LIMITATIONS: " + "; ".join(self.limitations))

        sheet.text = "\n".join(lines)
        sheet.advice = advice
        sheet.aggregates = agg
        sheet.allowed_names = [ven.name for ven in g.venues] + [self.place_label]
        return sheet
