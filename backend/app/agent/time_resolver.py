"""Deterministic resolution of time phrases into concrete local windows (no SLM date arithmetic).

All vocabulary comes from knowledge/time_lexicon.yaml. Sunrise/sunset anchored dayparts are left as anchors and
filled in later from the live Open-Meteo daily forecast (`apply_anchors`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from app.knowledge import get_knowledge


@dataclass
class Window:
    day: date
    start_h: float
    end_h: float
    label: str
    anchor: Optional[str] = None          # "sunrise" | "sunset"
    anchor_before: float = 0.0
    anchor_after: float = 0.0

    def start_dt(self) -> datetime:
        return datetime.combine(self.day, datetime.min.time()) + timedelta(hours=self.start_h)

    def end_dt(self) -> datetime:
        return datetime.combine(self.day, datetime.min.time()) + timedelta(hours=self.end_h)

    def to_dict(self) -> Dict:
        return {"day": self.day.isoformat(), "start": _fmt_h(self.start_h), "end": _fmt_h(self.end_h),
                "label": self.label}


@dataclass
class Resolution:
    windows: List[Window]
    compare: bool = False
    assumed: bool = False
    notes: List[str] = field(default_factory=list)
    explicit_hour: Optional[float] = None


def _fmt_h(h: float) -> str:
    h = max(0.0, min(23.99, h))
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


def _has(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", text) is not None


class TimeResolver:
    def __init__(self) -> None:
        self.lex = get_knowledge().time

    # ------------------------------------------------------------------ public
    def resolve(self, text: Optional[str], now: datetime) -> Resolution:
        t = (text or "").lower().strip()
        days, compare, notes = self._days(t, now)
        part, part_label = self._daypart(t)
        hour = self._explicit_hour(t)
        assumed = False

        if not days and not part and hour is None:
            dw = self.lex["default_window"]
            assumed = True
            notes.append(f"No time given: checked the next {dw['hours_ahead']} hours ({_fmt_h(dw['start'])}–{_fmt_h(dw['end'])} each day).")
            end = now + timedelta(hours=dw["hours_ahead"])
            d = now.date()
            windows = []
            while d <= end.date():
                s = max(dw["start"], now.hour + now.minute / 60 + 0.25) if d == now.date() else dw["start"]
                e = dw["end"] if d < end.date() else min(dw["end"], end.hour + end.minute / 60)
                if e - s >= 1:
                    windows.append(Window(d, s, e, "next hours"))
                d += timedelta(days=1)
            return Resolution(windows=windows, assumed=True, notes=notes)

        if not days:
            days = [now.date()]
            # a daypart that has already passed today rolls to tomorrow
            if part and not part.get("anchor") and not part.get("now") and now.hour + now.minute / 60 > part["end"] - 0.5:
                days = [now.date() + timedelta(days=1)]
                notes.append(f"{part_label.capitalize()} today has passed, so I looked at tomorrow.")
            elif hour is not None and now.hour + now.minute / 60 > hour:
                days = [now.date() + timedelta(days=1)]
                notes.append("That time has passed today, so I looked at tomorrow.")

        windows: List[Window] = []
        for d in days:
            if part and part.get("now"):
                start = now.hour + now.minute / 60
                windows.append(Window(d, start, min(23.5, start + part["hours"]), "now"))
            elif hour is not None:
                windows.append(Window(d, max(0.0, hour - 0.5), min(23.5, hour + 1.5), f"around {_fmt_h(hour)}"))
            elif part:
                if part.get("anchor"):
                    windows.append(Window(d, 0, 0, part_label, anchor=part["anchor"],
                                          anchor_before=part["before"], anchor_after=part["after"]))
                else:
                    windows.append(Window(d, part["start"], part["end"], part_label))
            else:
                dw = self.lex["default_window"]
                windows.append(Window(d, dw["start"], dw["end"], "daytime"))

        # trim today's windows to the future
        cur = now.hour + now.minute / 60
        trimmed = []
        for w in windows:
            if w.day == now.date() and not w.anchor:
                w.start_h = max(w.start_h, cur + 0.25)
                if w.end_h - w.start_h < 0.75:
                    continue
            trimmed.append(w)
        if not trimmed and windows:
            notes.append("The requested time has already passed.")
        return Resolution(windows=trimmed, compare=compare and len({w.day for w in trimmed}) > 1,
                          assumed=assumed, notes=notes, explicit_hour=hour)

    def apply_anchors(self, windows: List[Window], sun: Dict[date, Tuple[float, float]]) -> List[Window]:
        """Fill sunrise/sunset-anchored windows from live daily data: sun[day] = (sunrise_h, sunset_h)."""
        out = []
        for w in windows:
            if w.anchor and w.day in sun:
                base = sun[w.day][0] if w.anchor == "sunrise" else sun[w.day][1]
                w.start_h, w.end_h = base - w.anchor_before, base + w.anchor_after
            if not w.anchor or w.day in sun:
                out.append(w)
        return out

    def horizon_check(self, res: Resolution, now: datetime) -> List[str]:
        hz = self.lex["horizon_days"]
        notes = []
        last = max((w.day for w in res.windows), default=now.date())
        if (last - now.date()).days > hz["weather"]:
            notes.append("beyond_weather_horizon")
        elif (last - now.date()).days > hz["air_quality"]:
            notes.append("beyond_aq_horizon")
        return notes

    # ------------------------------------------------------------------ parts
    def _days(self, t: str, now: datetime) -> Tuple[List[date], bool, List[str]]:
        today = now.date()
        found: List[date] = []
        notes: List[str] = []
        compare = False

        m = re.search(r"(?:next|coming|following)\s+(\d+|two|three|four|five|couple of)\s+days", t)
        if m:
            words = {"two": 2, "three": 3, "four": 4, "five": 5, "couple of": 2}
            n = int(m.group(1)) if m.group(1).isdigit() else words[m.group(1)]
            start = today if now.hour < 20 else today + timedelta(days=1)
            found = [start + timedelta(days=i) for i in range(n)]
            compare = True

        if "weekend" in t:
            wk = self.lex["weekend_days"]
            offset = 7 if "next weekend" in t and today.weekday() not in wk else 0
            base = today + timedelta(days=offset)
            for wd in wk:
                delta = (wd - base.weekday()) % 7
                d = base + timedelta(days=delta)
                if "next weekend" in t and today.weekday() in wk:
                    d += timedelta(days=7)
                found.append(d)
            compare = True

        for phrase, offset in sorted(self.lex["relative_days"].items(), key=lambda kv: -len(kv[0])):
            if _has(t, phrase):
                d = today + timedelta(days=offset)
                if d not in found:
                    found.append(d)
                t = re.sub(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", " ", t)

        weekday_hits = []
        for name, wd in self.lex["weekdays"].items():
            if _has(t, name) and not (name == "sun" and _has(t, "sunset")) and wd not in weekday_hits:
                weekday_hits.append(wd)
        for wd in weekday_hits:
            delta = (wd - today.weekday()) % 7
            if re.search(r"next\s+" + "|".join(k for k, v in self.lex["weekdays"].items() if v == wd), t) and delta == 0:
                delta = 7
            d = today + timedelta(days=delta)
            if d not in found:
                found.append(d)
        if len(weekday_hits) >= 2 or (weekday_hits and " or " in t):
            compare = True

        months = self.lex["months"]
        for m in re.finditer(r"(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(months) + r")\b", t):
            day_n, mon = int(m.group(1)), months[m.group(2)]
            year = today.year + (1 if mon < today.month else 0)
            try:
                found.append(date(year, mon, day_n))
            except ValueError:
                notes.append(f"Couldn't read the date '{m.group(0)}'.")
        found = sorted(set(found))
        return found, compare or (" or " in t and len(found) > 1), notes

    def _daypart(self, t: str) -> Tuple[Optional[Dict], str]:
        for phrase, spec in sorted(self.lex["dayparts"].items(), key=lambda kv: (not kv[1].get("anchor"), -len(kv[0]))):
            if _has(t, phrase):
                return spec, phrase
        return None, ""

    @staticmethod
    def _explicit_hour(t: str) -> Optional[float]:
        m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", t)
        if m:
            h = int(m.group(1)) % 12 + (12 if m.group(3).startswith("p") else 0)
            return h + (int(m.group(2)) / 60 if m.group(2) else 0)
        m = re.search(r"\b(?:at|by|around)\s+([01]?\d|2[0-3]):([0-5]\d)\b", t)
        if m:
            return int(m.group(1)) + int(m.group(2)) / 60
        return None
