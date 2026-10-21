"""Generate the intent-extraction fine-tuning set (exact labels by construction). Runs in Colab.

    python training/gen_intent_data.py --n 3000 --out training/data/intent.jsonl

Vocabulary (activities, synonyms, profiles) comes from backend/app/knowledge so the labels match the runtime schema.
Place names below are TRAINING DATA ONLY (the runtime has no place lists). The eval gold set
(backend/eval/intent_gold.jsonl) and the hackathon's published example questions are excluded.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.knowledge import get_knowledge  # noqa: E402
from app.slm.prompts import intent_messages  # noqa: E402

PLACES = [
    "Aundh", "Baner", "Kothrud", "Viman Nagar", "Wakad", "Magarpatta", "Hadapsar", "Shivajinagar", "Camp", "Kalyani Nagar",
    "Goregaon", "Malad", "Borivali", "Kandivali", "Dadar", "Worli", "Colaba", "Chembur", "Mulund", "Vashi", "Nerul",
    "Belapur", "Thane West", "Ghodbunder Road", "Juhu", "Versova", "Santacruz", "Vile Parle", "Sion", "Matunga",
    "Indiranagar", "Koramangala", "HSR Layout", "Jayanagar", "JP Nagar", "Malleshwaram", "Hebbal", "Yelahanka",
    "Electronic City", "Marathahalli", "Sarjapur Road", "Bannerghatta Road", "Rajajinagar", "Basavanagudi",
    "Banjara Hills", "Jubilee Hills", "Gachibowli", "Madhapur", "Kondapur", "Secunderabad", "Begumpet", "Kukatpally",
    "Adyar", "Besant Nagar", "Anna Nagar", "T Nagar", "Velachery", "Mylapore", "OMR", "ECR", "Tambaram",
    "Salt Lake", "New Town", "Ballygunge", "Gariahat", "Howrah", "Dum Dum",
    "Connaught Place", "Hauz Khas", "Saket", "Dwarka", "Rohini", "Vasant Kunj", "Lajpat Nagar", "Mayur Vihar",
    "Noida Sector 18", "Greater Noida", "Gurugram", "DLF Phase 3", "Sohna Road", "Faridabad", "Ghaziabad",
    "Navrangpura", "Satellite", "Bodakdev", "SG Highway", "Vastrapur", "Maninagar",
    "Malviya Nagar Jaipur", "C Scheme", "Vaishali Nagar", "Mansarovar",
    "Gomti Nagar", "Hazratganj", "Aliganj", "Indira Nagar Lucknow",
    "Sector 17 Chandigarh", "Sukhna Lake", "Panchkula", "Mohali",
    "Kakkanad", "Fort Kochi", "Edappally", "Marine Drive Kochi", "Kowdiar", "Technopark",
    "Panaji", "Calangute", "Margao", "Anjuna", "Mapusa",
    "Bhopal", "Indore", "Vijay Nagar Indore", "Nagpur", "Dharampeth", "Nashik", "Aurangabad", "Kolhapur",
    "Mysuru", "Mangaluru", "Hubballi", "Coimbatore", "RS Puram", "Madurai", "Tiruchirappalli", "Visakhapatnam",
    "Vijayawada", "Guntur", "Bhubaneswar", "Cuttack", "Patna", "Ranchi", "Guwahati", "Shillong", "Dehradun",
    "Mussoorie", "Rishikesh", "Shimla", "Manali", "Udaipur", "Jodhpur", "Amritsar", "Ludhiana", "Varanasi", "Agra",
    "Ooty", "Kodaikanal", "Munnar", "Coorg", "Mahabaleshwar", "Matheran", "Alibaug", "Karjat", "Igatpuri",
    "Sinhagad", "Rajmachi", "Kalsubai", "Nandi Hills", "Skandagiri", "Tiger Hill", "Sanjay Van", "Aravalli Biodiversity Park",
]

TIMES = [
    "today", "this morning", "this evening", "tonight", "tomorrow", "tomorrow morning", "tomorrow evening",
    "tomorrow afternoon", "day after tomorrow", "this weekend", "next weekend", "on saturday", "on sunday",
    "saturday morning", "sunday evening", "this friday", "next monday", "in the next two days", "over the next 3 days",
    "after work", "before office", "at 6 am", "at 7:30 am", "around 5 pm", "at sunrise", "at sunset",
    "during golden hour", "on 2 nov", "on the 15th of november", "this week", "early morning", "at dawn",
]
HINGLISH_TIMES = ["kal subah", "kal shaam", "aaj shaam", "parso subah", "aaj raat", "kal dopahar"]

PROFILE_PHRASES = [
    ("I have asthma", ["asthma"], None), ("i'm asthmatic", ["asthma"], None), ("I get wheezy in pollution", ["asthma"], None),
    ("with my parents", ["senior"], None), ("my grandfather is 78", ["senior"], 78), ("for my 65 year old mom", ["senior"], 65),
    ("with two kids", ["child"], None), ("with my 6 year old", ["child"], 6), ("for a school group", ["child"], None),
    ("I'm 7 months pregnant", ["pregnancy"], None), ("my wife is pregnant", ["pregnancy"], None),
    ("I'm a heart patient", ["cardiac"], None), ("I have high blood pressure", ["cardiac"], None),
    ("I can't handle heat", ["heat_sensitive"], None), ("I'm recovering from knee surgery", ["mobility_limited"], None),
    ("I'm 30 and fit", [], 30), ("", [], None), ("", [], None), ("", [], None), ("", [], None), ("", [], None),
]
PREFS = [("somewhere quiet", ["quiet"]), ("with lots of shade", ["shade"]), ("near a lake", ["water"]),
         ("away from traffic", ["avoid_traffic"]), ("close by", ["close_by"]), ("with a nice view", ["scenic"]),
         ("", []), ("", []), ("", []), ("", [])]

TEMPLATES = [
    "{where_q} {act} {time} {loc_p}{prof}{pref}?",
    "best time {time} for {act} {loc_p}{radius}{prof}",
    "is {time} good for {act} {loc_p}{prof}",
    "{act} {loc_p} {time}, any suggestions{prof}{pref}",
    "suggest a spot for {act} {loc_p}{radius} {time}{prof}",
    "planning {act} {time} {loc_p}. what should I keep in mind{prof}?",
    "{prof_lead}want to go for {act} {time} {loc_p}{pref}",
    "can we do {act} {loc_p} {time}{travel}",
    "{act} {time}{travel} {loc_p}",
]
WHERE_Q = ["where can I go for", "where should I do", "good place for", "where to do", "best place for", "any place for"]
LOC_PREP = ["near {p}", "around {p}", "in {p}", "close to {p}", "within {r} km of {p}"]
OUT_OF_SCOPE = [
    "book me a flight to {p}", "best restaurants in {p}", "which stock should I buy today", "find an indoor gym in {p}",
    "how do I fix my laptop", "recommend a movie for tonight", "cheapest hotel in {p} this weekend",
    "write my college essay", "what is the capital of france", "order biryani near {p}", "bank holidays this month",
]
HINGLISH = ["{time} {act} karna hai {p} mein", "{p} ke paas {act} ke liye acchi jagah {time}?",
            "{time} {p} mein {act} safe hai kya", "mujhe {time} {act} karne jana hai {p} ke aas paas"]


def typo(s: str, rng: random.Random) -> str:
    if len(s) < 8 or rng.random() > 0.15:
        return s
    i = rng.randrange(1, len(s) - 1)
    return s[:i] + s[i + 1:] if rng.random() < 0.5 else s[:i] + s[i] + s[i:]


def empty_intent() -> dict:
    return {"in_scope": True, "out_of_scope_reason": None, "activity": None, "activity_raw": None,
            "location_text": None, "use_user_location": False, "radius_km": None, "max_travel_min": None,
            "time_text": None, "duration_min": None, "age": None, "profiles": [], "group_size": None,
            "preferences": [], "is_follow_up": False}


def sample(rng: random.Random, kb) -> tuple[str, dict, dict | None]:
    acts = list(kb.activities.values())
    a = rng.choice(acts)
    syn = rng.choice(a.synonyms)
    p = rng.choice(PLACES)
    label = empty_intent()
    r = rng.random()

    if r < 0.08:                                       # out of scope
        q = rng.choice(OUT_OF_SCOPE).format(p=p)
        label.update(in_scope=False, out_of_scope_reason="not an outdoor activity request",
                     location_text=p if p in q else None)
        return q, label, None
    if r < 0.17:                                       # Hinglish
        t = rng.choice(HINGLISH_TIMES)
        q = rng.choice(HINGLISH).format(time=t, act=syn, p=p)
        label.update(activity=a.id, activity_raw=syn, location_text=p, time_text=t)
        return q, label, None
    if r < 0.21:                                       # follow-up
        prev = empty_intent()
        prev.update(activity=a.id, activity_raw=syn, location_text=p, time_text="tomorrow morning")
        t = rng.choice(["what about sunday?", "and in the evening?", "something closer?", "what if I take my kids?",
                        "how about saturday instead"])
        label.update(is_follow_up=True)
        if "sunday" in t or "saturday" in t or "evening" in t:
            label["time_text"] = {"what about sunday?": "sunday", "and in the evening?": "in the evening",
                                  "how about saturday instead": "saturday"}.get(t)
        if "kids" in t:
            label["profiles"] = ["child"]
        if "closer" in t:
            label["preferences"] = ["close_by"]
        return t, label, prev

    time = rng.choice(TIMES)
    prof_txt, prof_ids, age = rng.choice(PROFILE_PHRASES)
    pref_txt, pref_ids = rng.choice(PREFS)
    radius = rng.choice([None, None, None, 5, 10, 15, 20, 40])
    travel = rng.choice([None, None, None, None, 15, 20, 30, 45])
    no_loc = rng.random() < 0.12
    near_me = no_loc and rng.random() < 0.6
    no_time = rng.random() < 0.08
    no_act = rng.random() < 0.05

    loc_p = ""
    if not no_loc:
        prep = rng.choice(LOC_PREP)
        if "{r}" in prep:
            radius = radius or rng.choice([5, 10, 15, 25])
            loc_p = prep.format(p=p, r=radius)
            radius_txt = ""
        else:
            loc_p = prep.format(p=p)
            radius_txt = f" within {radius} km" if radius else ""
    else:
        loc_p = rng.choice(["near me", "around me", "nearby", "from home"]) if near_me else ""
        radius_txt = f" within {radius} km" if radius else ""
    tmpl = rng.choice(TEMPLATES)
    act_txt = "something outdoors" if no_act else syn
    q = tmpl.format(where_q=rng.choice(WHERE_Q), act=act_txt, time="" if no_time else time, loc_p=loc_p,
                    prof=(", " + prof_txt) if prof_txt else "", prof_lead=(prof_txt + ", ") if prof_txt else "",
                    pref=(" " + pref_txt) if pref_txt else "", radius=radius_txt,
                    travel=f" under {travel} min drive" if travel else "")
    q = " ".join(q.split()).replace(" ,", ",").replace(" ?", "?")
    if "{pref}" not in tmpl:
        pref_ids = []
    if "{prof}" not in tmpl and "{prof_lead}" not in tmpl:
        prof_ids, age = [], None
    if "{travel}" not in tmpl:
        travel = None
    if "{radius}" not in tmpl and "{r}" not in loc_p and not (not no_loc and "within" in loc_p):
        radius = None if "within" not in q else radius
    if rng.random() < 0.3:
        q = q.lower()
    q = typo(q, rng)

    label.update(activity=None if no_act else a.id, activity_raw=None if no_act else syn,
                 location_text=None if no_loc else p, use_user_location=near_me,
                 radius_km=float(radius) if radius and f"{radius} km" in q else None,
                 max_travel_min=travel if travel and f"{travel} min" in q else None,
                 time_text=None if no_time else time, age=age, profiles=sorted(prof_ids), preferences=pref_ids)
    return q, label, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "training" / "data" / "intent.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    kb = get_knowledge()
    excluded = {json.loads(l)["q"].lower() for l in (ROOT / "backend" / "eval" / "intent_gold.jsonl").read_text().splitlines() if l.strip()}
    seen, rows = set(), []
    while len(rows) < args.n:
        q, label, prev = sample(rng, kb)
        if q.lower() in seen or q.lower() in excluded:
            continue
        seen.add(q.lower())
        msgs = intent_messages(q, prev)
        msgs.append({"role": "assistant", "content": json.dumps(label, ensure_ascii=False)})
        rows.append({"task": "intent", "messages": msgs})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} intent examples to {out}")


if __name__ == "__main__":
    main()
