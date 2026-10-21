"""Loads the YAML knowledge base (activities, profiles, rules, advice, time lexicon, geo parameters).

Everything tunable lives in these files so that no threshold, tag or keyword is hard-coded in logic.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

KNOWLEDGE_DIR = Path(__file__).parent


def _load(name: str) -> Dict[str, Any]:
    with open(KNOWLEDGE_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


@dataclass(frozen=True)
class ActivityProfile:
    id: str
    label: str
    synonyms: List[str]
    venues: List[str]
    exertion: str
    daylight: str
    default_duration_min: int
    default_radius_km: float
    min_area_m2: float
    comfort: Dict[str, float]
    wind_comfort_kmh: float
    limits: Dict[str, float]
    weights: Dict[str, float]
    setting: Dict[str, float]
    travel_tolerance: float = 1.0
    nominatim_phrases: tuple = ()
    require_sport_match: bool = False
    golden_hour: bool = False


@dataclass(frozen=True)
class PersonalProfile:
    id: str
    label: str
    triggers: List[str]
    limits: Dict[str, float] = field(default_factory=dict)
    weight_multipliers: Dict[str, float] = field(default_factory=dict)
    advice: List[str] = field(default_factory=list)
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    force_daylight: bool = False
    max_travel_factor: float = 1.0


class Knowledge:
    def __init__(self) -> None:
        raw = _load("activities.yaml")
        defaults = raw["defaults"]
        self.activities: Dict[str, ActivityProfile] = {}
        for act_id, spec in raw["activities"].items():
            merged = _deep_merge(defaults, spec)
            self.activities[act_id] = ActivityProfile(
                id=act_id,
                label=merged["label"],
                synonyms=merged.get("synonyms", []),
                venues=merged["venues"],
                exertion=merged["exertion"],
                daylight=merged["daylight"],
                default_duration_min=int(merged["default_duration_min"]),
                default_radius_km=float(merged["default_radius_km"]),
                min_area_m2=float(merged["min_area_m2"]),
                comfort=merged["comfort"],
                wind_comfort_kmh=float(merged["wind_comfort_kmh"]),
                limits=merged["limits"],
                weights=merged["weights"],
                setting=merged["setting"],
                travel_tolerance=float(merged.get("travel_tolerance", 1.0)),
                golden_hour=bool(merged.get("golden_hour", False)),
                nominatim_phrases=tuple(merged.get("nominatim_phrases", [])),
                require_sport_match=bool(merged.get("require_sport_match", False)),
            )

        self.profiles: Dict[str, PersonalProfile] = {
            pid: PersonalProfile(id=pid, **spec) for pid, spec in _load("profiles.yaml")["profiles"].items()
        }
        self.rules: List[Dict[str, Any]] = _load("rules.yaml")["rules"]
        self.advice: Dict[str, List[Dict[str, Any]]] = _load("advice.yaml")
        self.time: Dict[str, Any] = _load("time_lexicon.yaml")
        self.geo: Dict[str, Any] = _load("geo.yaml")
        self.scoring: Dict[str, Any] = _load("scoring.yaml")
        self.grounding: Dict[str, Any] = _load("grounding.yaml")
        self.language: Dict[str, Any] = _load("language.yaml")
        self.phrasing: Dict[str, Any] = _load("phrasing.yaml")

    def effective_limits(self, activity: str, profile_ids: List[str]) -> Dict[str, float]:
        """Activity limits tightened by every active personal profile (never loosened)."""
        limits = dict(self.activities[activity].limits)
        for pid in profile_ids:
            prof = self.profiles.get(pid)
            if not prof:
                continue
            for key, val in prof.limits.items():
                if key not in limits:
                    limits[key] = val
                elif key.endswith("_min"):
                    limits[key] = max(limits[key], val)
                else:
                    limits[key] = min(limits[key], val)
        return limits

    def effective_weights(self, activity: str, profile_ids: List[str]) -> Dict[str, float]:
        weights = dict(self.activities[activity].weights)
        for pid in profile_ids:
            prof = self.profiles.get(pid)
            if prof:
                for key, mult in prof.weight_multipliers.items():
                    if key in weights:
                        weights[key] *= mult
        return weights


@lru_cache()
def get_knowledge() -> Knowledge:
    return Knowledge()
