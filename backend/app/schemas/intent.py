"""Structured intent extracted from a free-form question (output of the SLM, or of the rule fallback)."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from app.knowledge import get_knowledge

ACTIVITY_IDS = list(get_knowledge().activities.keys())
PROFILE_IDS = list(get_knowledge().profiles.keys())


class Intent(BaseModel):
    in_scope: bool = Field(True, description="false if the request is not about doing an activity outdoors")
    out_of_scope_reason: Optional[str] = None
    activity: Optional[str] = Field(None, description=f"one of {ACTIVITY_IDS}, or null if unclear")
    activity_raw: Optional[str] = Field(None, description="the user's own words for the activity")
    location_text: Optional[str] = Field(None, description="place name exactly as written, without time words")
    use_user_location: bool = Field(False, description="true for 'near me', 'from home', 'to work' with no place")
    radius_km: Optional[float] = None
    max_travel_min: Optional[int] = None
    time_text: Optional[str] = Field(None, description="time expression copied verbatim, e.g. 'tomorrow evening'")
    duration_min: Optional[int] = None
    age: Optional[int] = None
    profiles: List[str] = Field(default_factory=list, description=f"subset of {PROFILE_IDS}")
    group_size: Optional[int] = None
    preferences: List[Literal["quiet", "shade", "water", "avoid_traffic", "close_by", "scenic"]] = Field(default_factory=list)
    is_follow_up: bool = False

    @classmethod
    def from_model_output(cls, data: dict) -> "Intent":
        """Build from SLM JSON, keeping every field that validates and dropping only the invalid ones (a model
        without grammar-constrained decoding may emit e.g. preferences=['wind'])."""
        from pydantic import ValidationError
        clean = {k: v for k, v in (data or {}).items() if k in cls.model_fields}
        allowed_prefs = set(cls.model_fields["preferences"].annotation.__args__[0].__args__)
        if isinstance(clean.get("preferences"), list):
            clean["preferences"] = [p for p in clean["preferences"] if p in allowed_prefs]
        if isinstance(clean.get("profiles"), list):
            clean["profiles"] = [p for p in clean["profiles"] if isinstance(p, str)]
        for _ in range(len(clean) + 1):
            try:
                return cls(**clean)
            except ValidationError as exc:
                bad = {str(e["loc"][0]) for e in exc.errors() if e.get("loc")}
                if not bad:
                    raise
                for k in bad:
                    clean.pop(k, None)
        return cls()

    def normalised(self) -> "Intent":
        kb = get_knowledge()
        data = self.model_dump()
        if data["activity"] not in kb.activities:
            data["activity"] = None
        data["profiles"] = sorted({p for p in data["profiles"] if p in kb.profiles})
        if data["age"] is not None:
            for pid, prof in kb.profiles.items():
                if prof.min_age and data["age"] >= prof.min_age and pid not in data["profiles"]:
                    data["profiles"].append(pid)
                if prof.max_age and data["age"] <= prof.max_age and pid not in data["profiles"]:
                    data["profiles"].append(pid)
        if data["radius_km"] is not None:
            from app.config import get_settings
            data["radius_km"] = max(0.5, min(float(data["radius_km"]), get_settings().MAX_RADIUS_KM))
        if data["location_text"]:
            data["location_text"] = data["location_text"].strip(" ,.?!") or None
        if data["location_text"]:
            # a "location" made only of time/filler words ("right now", "this evening") is not a place
            lex = kb.time
            timeish = set(lex["dayparts"]) | set(lex["relative_days"]) | set(lex["weekdays"]) | set(lex["filler_words"])
            if all(tok in timeish for tok in data["location_text"].lower().split()):
                data["location_text"] = None
        return Intent(**data)


def intent_json_schema() -> dict:
    """JSON schema handed to llama.cpp for grammar-constrained decoding."""
    schema = Intent.model_json_schema()
    schema["properties"]["activity"] = {"anyOf": [{"type": "string", "enum": ACTIVITY_IDS}, {"type": "null"}]}
    schema["properties"]["profiles"] = {"type": "array", "items": {"type": "string", "enum": PROFILE_IDS}}
    return schema
