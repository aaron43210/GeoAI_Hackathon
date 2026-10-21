"""Rule-based intent extraction: the fallback when the SLM is unavailable, and a cross-check for its output.

Vocabulary comes from the knowledge base (activity synonyms, profile triggers, time lexicon). There are no
place lists: the location is extracted as the phrase after a spatial preposition, stripped of time words, and is
only accepted later if the geocoder resolves it.
"""
from __future__ import annotations

import re
from typing import List, Optional

from app.knowledge import get_knowledge
from app.schemas.intent import Intent

# Spatial cues, strongest first. Hinglish postpositions ("<place> mein", "<place> ke paas") come after the place.
_PREPOSITIONS = [(r"within \d+(?:\.\d+)?\s*(?:km|kms)\s+of", 4), (r"near|around|close to|nearby|next to", 4),
                 (r"in|at", 3), (r"from|of|towards|to", 1)]
_POSTPOSITIONS = r"(?:mein|me|ke paas|ke pass|ke aas paas|ke najdik|madhe|madhye|alli)"
_STOP_WORDS = {"me", "my", "home", "work", "office", "the", "a", "an", "here", "us", "it", "this", "that", "to",
               "next", "coming", "go", "do", "be", "place", "places", "spot", "spots", "kids", "parents", "okay",
               "ok", "safe", "good", "best", "outdoors", "outdoor", "outside", "any", "some", "area", "city",
               # Hinglish function words
               "karna", "karne", "hai", "hain", "ke", "ka", "ki", "ko", "jana", "jaana", "mujhe", "kya", "kahan",
               "kab", "acha", "accha", "liye", "chahiye", "tha", "thi", "khelna", "khelne", "khelenge", "khel",
               "chalna", "chalne", "ghoomna", "ghumne", "ghumna", "daudna", "daudne", "karenge", "karte", "jayenge",
               "jaenge", "hum", "main", "mai", "apne", "saath"}
OUT_OF_SCOPE_HINTS = ["indoor", "gym", "netflix", "movie", "flight", "stock", "recipe", "bank", "crypto", "code",
                      "homework", "loan", "hotel booking", "book a cab"]


def _time_words() -> List[str]:
    lex = get_knowledge().time
    words = list(lex["dayparts"]) + list(lex["relative_days"]) + list(lex["weekdays"]) + list(lex["months"])
    words += ["weekend", "this", "next", "coming", "days", "day", "week", "after", "before", "work", "evening", "today",
              "within", "km", "minutes", "min", "hours", "hour", "for", "with", "and", "or", "is", "it", "safe", "ok"]
    return sorted(set(words), key=len, reverse=True)


def detect_activity(q: str) -> tuple[Optional[str], Optional[str]]:
    best = (None, None, 0)
    for act_id, prof in get_knowledge().activities.items():
        for syn in prof.synonyms + [act_id.replace("_", " ")]:
            if re.search(r"(?<![a-z])" + re.escape(syn) + r"(?![a-z])", q) and len(syn) > best[2]:
                best = (act_id, syn, len(syn))
    return best[0], best[1]


def _clean_candidate(chunk: str, junk: set) -> Optional[str]:
    chunk = re.split(r"[,?!.;]| (?:for|with|on|at|by|this|next|tomorrow|today|tonight|in the|during|after|before|"
                     r"between|when|where|and|or|is|if)\b", chunk, maxsplit=1, flags=re.IGNORECASE)[0]
    tokens = [tok for tok in chunk.strip().split() if tok.lower().strip("'") not in junk]
    cand = " ".join(tokens).strip(" '")
    return cand if len(cand) > 2 else None


def detect_location(question: str) -> Optional[str]:
    kb = get_knowledge()
    junk = set(_time_words()) | _STOP_WORDS
    for prof in kb.activities.values():
        junk.update(syn.lower() for syn in prof.synonyms if " " not in syn)
    found = []
    for pattern, strength in _PREPOSITIONS:
        for m in re.finditer(r"(?<![a-z])(?:" + pattern + r")\s+([A-Za-z][\w'.\- ]{1,60})", question, flags=re.IGNORECASE):
            cand = _clean_candidate(m.group(1), junk)
            if cand:
                found.append((strength, m.start(), cand))
    for m in re.finditer(r"((?:[A-Za-z][\w'.\-]*\s+){1,3})" + _POSTPOSITIONS + r"(?![a-z])", question, flags=re.IGNORECASE):
        cand = _clean_candidate(m.group(1), junk)
        if cand:
            found.append((3, m.start(), cand))
    if not found:
        return None
    found.sort(key=lambda f: (-f[0], f[1]))
    return found[0][2]


def parse(question: str) -> Intent:
    kb = get_knowledge()
    q = question.lower()
    activity, raw = detect_activity(q)

    profiles = [pid for pid, prof in kb.profiles.items()
                if any(re.search(r"(?<![a-z])" + re.escape(trig) + r"(?![a-z])", q) for trig in prof.triggers)]
    age = None
    m = re.search(r"\b(\d{1,3})\s*(?:-?\s*year[- ]?old|yrs?\b|years old|yo\b)", q)
    if m:
        age = int(m.group(1))
    else:
        m = re.search(r"\b(?:is|aged|age|turning)\s+(\d{1,3})\b(?!\s*(?:km|kms|min|mins|minutes|am|pm|%|°|hours?|hrs?|people|guests))", q)
        if m and 4 <= int(m.group(1)) <= 110:
            age = int(m.group(1))

    radius = None
    m = re.search(r"(?:within|under|less than|upto|up to)?\s*(\d+(?:\.\d+)?)\s*(?:km|kms|kilomet)", q)
    if m:
        radius = float(m.group(1))
    travel = None
    m = re.search(r"(?:within|under|less than|max|upto|up to)\s*(\d{1,3})\s*(?:min|mins|minutes)\s*(?:drive|away|travel|ride)?", q)
    if m:
        travel = int(m.group(1))
    duration = None
    m = re.search(r"(?:for|about)\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\b", q)
    if m:
        duration = int(float(m.group(1)) * 60)
    m = re.search(r"(?:for|about)\s+(\d{2,3})\s*(?:min|mins|minutes)\b", q)
    if m and duration is None:
        duration = int(m.group(1))

    location = detect_location(question)
    use_user_location = bool(re.search(r"near me|nearby|around me|from home|to work|to office|my area|close to me", q)) and not location

    prefs = []
    for pref, pat in {"quiet": r"quiet|peaceful|less crowd", "shade": r"shade|shady|trees", "water": r"lake|river|sea|beach|water",
                      "avoid_traffic": r"traffic|pollution|clean air", "close_by": r"close by|nearby|short drive",
                      "scenic": r"scenic|view|beautiful"}.items():
        if re.search(pat, q):
            prefs.append(pref)

    in_scope = activity is not None or not any(h in q for h in OUT_OF_SCOPE_HINTS)
    return Intent(
        in_scope=in_scope,
        out_of_scope_reason=None if in_scope else "not an outdoor activity request",
        activity=activity,
        activity_raw=raw,
        location_text=location,
        use_user_location=use_user_location,
        radius_km=radius,
        max_travel_min=travel,
        time_text=question,              # the resolver scans the full question for time expressions
        duration_min=duration,
        age=age,
        profiles=profiles,
        preferences=prefs,
    ).normalised()
