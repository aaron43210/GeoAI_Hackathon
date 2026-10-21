"""Prompt templates. The same templates are used to build the fine-tuning data (training/), so inference and
training stay aligned. Few-shot examples deliberately avoid the hackathon's published example queries."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.knowledge import get_knowledge


def intent_system() -> str:
    kb = get_knowledge()
    acts = ", ".join(f"{a} ({p.label})" for a, p in kb.activities.items())
    profs = ", ".join(f"{p} ({v.label})" for p, v in kb.profiles.items())
    return f"""You extract structured fields from a question about doing something outdoors. Reply with JSON only.
Fields:
- in_scope: false only if the request is not about going outdoors for an activity (e.g. booking, indoor, finance).
- activity: one of [{acts}] or null if unclear. "walk" is walking, "jog" is running, "trek" is hiking, a party/wedding is outdoor_event.
- activity_raw: the user's words for the activity.
- location_text: the place name exactly as written (fix obvious typos only). No time words. null if none.
- use_user_location: true if they say near me / from home / to work without naming a place.
- radius_km, max_travel_min, duration_min, age, group_size: numbers if stated, else null. radius_km is a search
  distance ("within 10 km of X"), never the length of a run or ride ("a 5 km run" has no radius).
- time_text: copy the time expression verbatim (e.g. "tomorrow evening", "this weekend", "kal subah"). null if none. Never compute dates.
- profiles: any of [{profs}] that apply to the people going ("my parents" -> senior, "kids" -> child).
- preferences: any of quiet, shade, water, avoid_traffic, close_by, scenic if asked for.
- is_follow_up: true if the message only changes something about the previous request."""


FEW_SHOT = [
    ("any quiet spot for yoga near Aundh day after tomorrow morning? my wife is 6 months pregnant",
     {"in_scope": True, "out_of_scope_reason": None, "activity": "yoga", "activity_raw": "yoga", "location_text": "Aundh",
      "use_user_location": False, "radius_km": None, "max_travel_min": None, "time_text": "day after tomorrow morning",
      "duration_min": None, "age": None, "profiles": ["pregnancy"], "group_size": 2, "preferences": ["quiet"],
      "is_follow_up": False}),
    ("can u suggest where to book a cheap hotel in goa",
     {"in_scope": False, "out_of_scope_reason": "hotel booking is not an outdoor activity request", "activity": None,
      "activity_raw": None, "location_text": "goa", "use_user_location": False, "radius_km": None, "max_travel_min": None,
      "time_text": None, "duration_min": None, "age": None, "profiles": [], "group_size": None, "preferences": [],
      "is_follow_up": False}),
    ("5-a-side under 20 min drive from me this fri night",
     {"in_scope": True, "out_of_scope_reason": None, "activity": "football", "activity_raw": "5-a-side",
      "location_text": None, "use_user_location": True, "radius_km": None, "max_travel_min": 20,
      "time_text": "this fri night", "duration_min": None, "age": None, "profiles": [], "group_size": None,
      "preferences": [], "is_follow_up": False}),
]


def intent_messages(question: str, previous: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": intent_system()}]
    for q, a in FEW_SHOT:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": json.dumps(a, ensure_ascii=False)})
    content = question
    if previous:
        content = f"Previous request: {json.dumps(previous, ensure_ascii=False)}\nNew message: {question}"
    msgs.append({"role": "user", "content": content})
    return msgs


COMPOSE_SYSTEM = """You are TerraPulse, an outdoor-activity advisor. Write the answer to the user's question using ONLY the facts in the FACT SHEET.
Rules:
- Every number, time, date and place name you write must appear in the fact sheet. Never add weather, places or numbers of your own.
- Put the fact's id in square brackets right after it, like "feels like 29 °C [E4]".
- If a fact is marked unavailable, say it is unavailable; do not guess it.
- Use this structure with short markdown:
{structure}
**Why:** 2-4 bullets (weather, air, wind/roads/industry, travel).
**Carry:** bullets. **Watch for:** bullets. **Avoid:** bullets (times to avoid and why).
**Other options:** 1-2 bullets.
- Never call a value good if the fact sheet flags it. If there is a HEALTH WARNING line, state it plainly right after the best pick (do not soften it).
- Only say there are no other options if the fact sheet lists none.
- Under 220 words. Plain, friendly, specific to this person."""


STRUCTURE_OK = "**Best pick:** place, day and time window, one-sentence reason."
STRUCTURE_NOT_RECOMMENDED = ("**Not recommended:** say clearly it is not advisable in this period and the main reason, "
                             "then the least-bad option (place and time) and how to reduce the risk.")


def compose_messages(fact_sheet: str) -> List[Dict[str, str]]:
    structure = STRUCTURE_NOT_RECOMMENDED if "NOT RECOMMENDED" in fact_sheet else STRUCTURE_OK
    return [{"role": "system", "content": COMPOSE_SYSTEM.replace("{structure}", structure)},
            {"role": "user", "content": fact_sheet}]


REPAIR_NOTE = ("Your previous answer contained facts that are not in the fact sheet: {violations}. Rewrite it using "
               "only fact-sheet values and ids.")


EXPLAIN_SYSTEM = """You are TerraPulse, an outdoor-activity advisor, answering a follow-up question about your previous recommendation.
Use ONLY the facts in the FACT SHEET of that recommendation. Cite fact ids in square brackets like [E4].
Every number, time and place you mention must appear in the fact sheet. If the fact sheet does not contain the answer,
say so plainly. Never weaken a safety warning or a 'not recommended' decision. Answer in 2-5 short sentences."""


def explain_messages(fact_sheet: str, question: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": EXPLAIN_SYSTEM},
            {"role": "user", "content": f"FACT SHEET:\n{fact_sheet}\n\nFOLLOW-UP QUESTION: {question}"}]
