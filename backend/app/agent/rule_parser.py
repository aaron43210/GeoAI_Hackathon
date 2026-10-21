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

def _lang() -> dict:
    return get_knowledge().language


def expand_abbreviations(text: str) -> str:
    """Whole-word shorthand expansion ("nr cubbon prk" → "near cubbon park"); vocabulary in language.yaml."""
    for short, full in sorted(_lang()["abbreviations"].items(), key=lambda kv: -len(kv[0])):
        text = re.sub(r"(?<![\w])" + re.escape(short) + r"(?![\w])", full, text, flags=re.IGNORECASE)
    return text


def _time_words() -> List[str]:
    lex = get_knowledge().time
    words = list(lex["dayparts"]) + list(lex["relative_days"]) + list(lex["weekdays"]) + list(lex["months"])
    words += list(lex.get("filler_words", []))
    return sorted(set(words), key=len, reverse=True)


def detect_activity(q: str) -> tuple[Optional[str], Optional[str]]:
    best = (None, None, 0)
    for act_id, prof in get_knowledge().activities.items():
        for syn in prof.synonyms + [act_id.replace("_", " ")]:
            if re.search(r"(?<![a-z])" + re.escape(syn) + r"(?![a-z])", q) and len(syn) > best[2]:
                best = (act_id, syn, len(syn))
    return best[0], best[1]


def has_in_scope_cue(question: str) -> bool:
    q = question.lower()
    return any(re.search(r"(?<![a-z])" + re.escape(c) + r"(?![a-z])", q) for c in _lang()["in_scope_cues"])


def _clean_candidate(chunk: str, junk: set) -> Optional[str]:
    ends = "|".join(re.escape(w) for w in _lang()["location_terminators"])
    chunk = re.split(r"[,?!.;]| (?:" + ends + r")\b", chunk, maxsplit=1, flags=re.IGNORECASE)[0]
    tokens = [tok for tok in chunk.strip().split() if tok.lower().strip("'") not in junk]
    cand = " ".join(tokens).strip(" '")
    return cand if len(cand) > 2 else None


def _junk_words() -> set:
    junk = set(_time_words()) | set(_lang()["stop_words"])
    for prof in get_knowledge().activities.values():
        junk.update(syn.lower() for syn in prof.synonyms if " " not in syn)
    return junk


def tail_place_candidates(question: str) -> List[str]:
    """Trailing words that could be a place when no spatial cue exists ("… kal shaam dwarka delhi").
    Longest first; the pipeline accepts one only if the live geocoder resolves it to a settlement."""
    junk = _junk_words()
    tokens = re.findall(r"[A-Za-z][\w'.\-]*", expand_abbreviations(question))
    run: List[str] = []
    for tok in reversed(tokens):
        if tok.lower() in junk:
            if run:
                break
            continue
        run.insert(0, tok)
        if len(run) >= _lang()["tail_place_max_words"]:
            break
    return [" ".join(run[i:]) for i in range(len(run))] if run else []


def detect_location(question: str) -> Optional[str]:
    kb = get_knowledge()
    junk = _junk_words()
    found = []
    for prep in _lang()["prepositions"]:
        pattern, strength = prep["pattern"], prep["strength"]
        # lookahead so matches can overlap: "to work at 8 AM from Wakad" must still yield "from Wakad"
        for m in re.finditer(r"(?<![a-z])(?=(?:" + pattern + r")\s+([A-Za-z][\w'.\- ]{1,60}))", question,
                             flags=re.IGNORECASE):
            cand = _clean_candidate(m.group(1), junk)
            if cand:
                found.append((strength, m.start(), cand))
    post = "(?:" + "|".join(re.escape(p) for p in _lang()["postpositions"]) + ")"
    for m in re.finditer(r"((?:[A-Za-z][\w'.\-]*\s+){1,3})" + post + r"(?![a-z])", question, flags=re.IGNORECASE):
        cand = _clean_candidate(m.group(1), junk)
        if cand:
            found.append((3, m.start(), cand))
    if not found:
        return None
    found.sort(key=lambda f: (-f[0], f[1]))
    return found[0][2]


def parse(question: str) -> Intent:
    kb = get_knowledge()
    question = expand_abbreviations(question)
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
        lo, hi = _lang()["age_bounds"]
        if m and lo <= int(m.group(1)) <= hi:
            age = int(m.group(1))

    radius = None
    # a search radius needs a spatial cue: "within 15 km", "15 km radius", "10 km from/of/around" (not "a 5 km run")
    m = re.search(r"(?:within|under|less than|upto|up to|radius of)\s*(\d+(?:\.\d+)?)\s*(?:km|kms|kilomet)", q) or \
        re.search(r"(\d+(?:\.\d+)?)\s*(?:km|kms|kilomet\w*)\s+(?:radius|of|from|around|away)", q)
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

    in_scope = activity is not None or not any(
        re.search(r"(?<![a-z])" + re.escape(h) + r"(?![a-z])", q) for h in _lang()["out_of_scope_hints"])
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
