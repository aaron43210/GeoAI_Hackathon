"""Friendly chat answers: plain language first, then "The data behind this" with numbered sources.

Every value shown comes from the evidence ledger (the same facts the verifier checks). Inline references are short
footnote numbers [1], [2] that point to the data section at the end, where each fact is listed with its value,
source and time. Vocabulary (AQI categories, comfort words, openers) lives in knowledge/phrasing.yaml.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.geospatial.context import compass
from app.knowledge import get_knowledge

_CITE = re.compile(r"\[(E\d+)\]")


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return str(v)


def band(value: float, bands: List[Dict[str, Any]]) -> str:
    for b in bands:
        if value <= b["max"]:
            return b["word"]
    return bands[-1]["word"]


def _clock(dt: datetime) -> str:
    h = dt.hour % 12 or 12
    return f"{h}" if dt.minute == 0 else f"{h}:{dt.minute:02d}"


def human_window(start: datetime, end: datetime, now: datetime) -> str:
    """'tomorrow 6–8 am', 'on Saturday 11 am–2 pm', 'today 7–8 pm'."""
    words = get_knowledge().phrasing["day_words"]
    delta = (start.date() - now.date()).days
    day = words.get(delta) or f"on {start:%A}"
    s_ampm, e_ampm = ("am" if start.hour < 12 else "pm"), ("am" if end.hour < 12 or end.hour == 24 else "pm")
    span = f"{_clock(start)}–{_clock(end)} {e_ampm}" if s_ampm == e_ampm else \
        f"{_clock(start)} {s_ampm}–{_clock(end)} {e_ampm}"
    return f"{day} {span}"


@dataclass
class Footnotes:
    """Maps evidence ids to footnote numbers and renders the data section."""
    evidence: Dict[str, Dict[str, Any]]
    order: List[str] = field(default_factory=list)

    def ref(self, eid: Optional[str]) -> str:
        if not eid or eid not in self.evidence:
            return ""
        if eid not in self.order:
            self.order.append(eid)
        return f" [{self.order.index(eid) + 1}]"

    def convert(self, text: str) -> str:
        return re.sub(r"\s?\[(E\d+)\]", lambda m: self.ref(m.group(1)), text)

    @staticmethod
    def describe(e: Dict[str, Any]) -> str:
        v = e.get("value")
        if isinstance(v, (list, tuple)) and len(v) == 2:
            val = _fmt(v[0]) if v[0] == v[1] else f"{_fmt(v[0])}–{_fmt(v[1])}"
        elif isinstance(v, dict) and "from_deg" in v:
            val = f"from {compass(v['from_deg'])} at {_fmt(v['kmh'])}"
        elif isinstance(v, dict) and "sunrise" in v:
            val = f"sunrise {v['sunrise']}, sunset {v['sunset']}"
        else:
            val = _fmt(v)
        label = str(e.get("label") or "")
        if label.endswith(" up to"):
            label, val = label[:-len(" up to")], f"up to {val}"
        unit = e.get("unit") or ""
        unit = "" if unit == "0–1" else unit
        shown = f"{val}{'' if not unit else (unit if unit == '/100' else ' ' + unit)}"
        source = f"[{e.get('source')}]({e['url']})" if e.get("url") else str(e.get("source"))
        kind = {"forecast": "forecast", "observation": "observed", "modelled": "modelled estimate",
                "heuristic": "estimate", "computed": "computed"}.get(e.get("kind"), "")
        meta = ", ".join(x for x in (kind, e.get("time")) if x)
        if label == str(v):                       # a place: no "X: X"
            return f"**{label}** · {source}"
        return f"{label}: **{shown}** · {source}" + (f" ({meta})" if meta else "")

    def section(self) -> str:
        if not self.order:
            return ""
        ph = get_knowledge().phrasing
        lines = [f"[{i + 1}] {self.describe(self.evidence[eid])}" for i, eid in enumerate(self.order)]
        return f"---\n**{ph['data_heading']}**\n" + "\n".join(lines) + f"\n\n<sub>{ph['checked_note']}</sub>"


@dataclass
class FriendlyParts:
    head: str
    facts: str          # conditions in plain words (replaced by the model's explanation when it passes the verifier)
    actions: str        # take / watch for / skip / alternatives (always code-rendered)
    tail: str
    notes: Footnotes

    def compose(self, middle: Optional[str] = None) -> str:
        parts = [self.head, middle if middle else self.facts, self.actions, self.tail]
        text = "\n\n".join(p for p in parts if p)
        data = self.notes.section()
        return f"{text}\n\n{data}" if data else text


def render_friendly(grid, sheet, act, place_label: str, now: datetime, limitations: List[str]) -> FriendlyParts:
    ph = get_knowledge().phrasing
    ev = {e.id: e.to_dict() for e in sheet.ledger.items}
    fn = Footnotes(ev)
    focus = grid.best or grid.least_bad
    activity = act.label.split(" (")[0].split(" / ")[0].lower()
    if focus is None:
        return FriendlyParts("I couldn't find any forecast hours for that time, so I can't recommend anything yet.",
                             "", "", ph["next_steps"]["not_recommended"], fn)
    v = grid.venues[focus.venue]
    agg = sheet.aggregates or {}
    key = lambda name: sheet.ledger.get(name)  # noqa: E731
    k = lambda var, a: key(f"{var}:{a}:{v.id}:{focus.start}")  # noqa: E731
    when = human_window(focus.start, focus.end, now)
    venue_id = key(f"venue:{v.id}")

    def with_ref(text: str) -> str:
        bold = f"**{v.name}**"
        return text.replace(bold, bold + fn.ref(venue_id), 1) if bold in text else text + fn.ref(venue_id)

    severe = [h for h in focus.concerns if h.severity == "high"]
    if grid.best is None:
        outcome = "not_recommended"
    else:
        outcome = "caution" if (severe or focus.concerns) else "ok"

    head: List[str] = []
    if outcome == "not_recommended":
        head.append(ph["openers"]["not_recommended"].format(activity=activity, area=place_label))
        reasons = []
        for c in focus.cells:
            for h in c.hits:
                if h.decision == "avoid" and h.state.value == "true" and h.text not in reasons:
                    reasons.append(h.text)
        if reasons:
            head.append("Every hour you asked about hits a safety limit, mainly: " + "; ".join(reasons[:2]) + ".")
        head.append(with_ref(ph["least_bad"].format(venue=v.name, when=when)))
    else:
        head.append(with_ref(ph["openers"][outcome].format(venue=v.name, when=when, activity=activity)))

    aqi = agg.get("max_us_aqi")
    if severe and aqi is not None:
        aqi_ref = fn.ref(k("us_aqi", "max"))
        dom = _dominant_pollutant(focus)
        head.append(f"⚠️ **Heads-up:** the air will be **{band(aqi, ph['aqi_bands'])}** (US AQI up to {_fmt(aqi)}{aqi_ref}"
                    + (f", mostly {dom}" if dom else "") + "). Keep the effort light and short, and wear an N95 mask.")
    elif severe:
        head.append("⚠️ **Heads-up:** " + "; ".join(h.text for h in severe) + ".")

    body: List[str] = []
    cond = []
    if agg.get("max_apparent") is not None:
        ref = fn.ref(k("apparent_temperature", "range"))
        cond.append(f"**{band(agg['max_apparent'], ph['feels_bands'])}**, feels like about {round(agg['max_apparent'])} °C{ref}")
    if agg.get("max_precip_prob") is not None:
        ref = fn.ref(k("precipitation_probability", "max"))
        cond.append(f"**{band(agg['max_precip_prob'], ph['rain_bands'])}**, rain chance up to "
                    f"{_fmt(agg['max_precip_prob'])}%{ref}")
    if agg.get("max_gust") is not None:
        ref = fn.ref(k("wind_gusts_10m", "max"))
        cond.append(f"**{band(agg['max_gust'], ph['wind_bands'])}**, gusts up to {_fmt(agg['max_gust'])} km/h{ref}")
    if cond:
        body.append("**Weather:** " + " · ".join(cond))
    if aqi is not None and not severe:
        aqi_ref = fn.ref(k("us_aqi", "max"))
        body.append(f"**Air:** **{band(aqi, ph['aqi_bands'])}**, US AQI up to {_fmt(aqi)}{aqi_ref}")
    elif aqi is None:
        miss = key(f"aq-missing:{v.id}")
        body.append(f"I couldn't check air quality this time{fn.ref(miss)}.")
    sun_ref = fn.ref(key(f"sun:{v.id}:{focus.start.date()}")) if (
        act.golden_hour or (agg.get("minutes_to_sunset_after_end") is not None
                            and agg["minutes_to_sunset_after_end"] < 60)) else ""
    if sun_ref and agg.get("sunset"):
        hh, mm = (int(x) for x in agg["sunset"].split(":"))
        body.append(f"Sunset is at {_clock(datetime(2000, 1, 1, hh, mm))} pm{sun_ref}." if hh >= 12 else
                    f"Sunset is at {agg['sunset']}{sun_ref}.")
    expo_id = key(f"expo:{v.id}:{focus.start}")
    if expo_id and isinstance(ev[expo_id]["value"], (int, float)) and \
            ev[expo_id]["value"] >= get_knowledge().geo["exposure_significant"]:
        src = next((c.exposure.dominant_source for c in focus.cells if c.exposure and c.exposure.dominant_source), None)
        body.append(f"The wind carries some traffic/industry air from {src or 'nearby roads'} towards it then"
                    f"{fn.ref(expo_id)}.")
    if v.travel and focus.cells[0].travel_min is not None:
        rush_ref = fn.ref(key(f"rush:{v.id}:{focus.start}"))
        body.append(f"It's about {_fmt(focus.cells[0].travel_min)} min away at that time{rush_ref}.")

    actions: List[str] = []
    if outcome != "not_recommended":
        for r in grid.avoid_ranges[:2]:
            reason = r["reasons"][0].text if r["reasons"] else "a safety limit"
            ref = fn.ref(key("avoid:" + str(r["start"])))
            actions.append(f"Skip {human_window(r['start'], r['end'], now)}: {reason}{ref}.")

    advice = sheet.advice or {}
    if advice.get("carry"):
        actions.append("**Take:** " + ", ".join(i["text"] for i in advice["carry"]) + ".")
    watch = [i["text"] for i in advice.get("watch_for", []) if not i["text"].startswith("sunset at")]
    if watch:
        actions.append("**Watch for:** " + "; ".join(watch) + ".")

    alts = []
    for a in grid.alternatives:
        av = grid.venues[a.venue]
        mins = f", {_fmt(av.travel['minutes'])} min away" if av.travel else ""
        alt_ref = fn.ref(key(f"venue:{av.id}"))
        alts.append(f"{av.name} ({human_window(a.start, a.end, now)}{mins}){alt_ref}")
    if alts and outcome != "not_recommended":
        actions.append("**Also good:** " + "; ".join(alts) + ".")
    if len(grid.compare) > 1:
        days = [d for d in grid.compare if d["best"]]
        if len(days) > 1:
            best_day = max(days, key=lambda d: d["best"].score)
            actions.append(f"Comparing days, **{best_day['day']:%A}** looks better.")

    notes = [n for n in limitations if "unavailable" in n or "not checked" in n.lower()][:1]
    tail = []
    if notes:
        tail.append(f"<sub>Note: {notes[0]}.</sub>")
    tail.append(ph["next_steps"][outcome])
    return FriendlyParts("\n".join(head), "\n".join(body), "\n".join(actions), "\n\n".join(tail), fn)


def _dominant_pollutant(pick) -> Optional[str]:
    names = get_knowledge().phrasing["pollutant_names"]
    worst = max((c for c in pick.cells if c.vars.get("us_aqi") is not None), key=lambda c: c.vars["us_aqi"], default=None)
    if worst is None:
        return None
    subs = {"PM2.5": worst.vars.get("us_aqi_pm2_5"), "PM10": worst.vars.get("us_aqi_pm10"),
            "ozone": worst.vars.get("us_aqi_ozone"), "NO2": worst.vars.get("us_aqi_nitrogen_dioxide")}
    subs = {k: x for k, x in subs.items() if x is not None}
    return names.get(max(subs, key=subs.get)) if subs else None


def friendly_explanation(text: str, evidence: List[Dict[str, Any]]) -> str:
    """Turn a cited explanation into chat text + the data section."""
    fn = Footnotes({e["id"]: e for e in evidence or []})
    body = fn.convert(text)
    body = re.sub(r"^- (RESULT|BEST|HEALTH WARNING): ", "- ", body, flags=re.M)
    data = fn.section()
    return f"{body}\n\n{data}" if data else body
