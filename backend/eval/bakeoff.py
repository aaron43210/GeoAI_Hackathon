"""Model bake-off: intent-extraction accuracy, grounded-composition rate and CPU latency for candidate GGUFs.

Usage:  python -m eval.bakeoff /path/a.gguf /path/b.gguf [--compose]
Writes eval/results/bakeoff.json. The gold set (eval/intent_gold.jsonl) deliberately avoids the hackathon's
published example questions.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from app.schemas.intent import Intent, intent_json_schema
from app.slm.prompts import compose_messages, intent_messages
from app.slm.runtime import render_chatml, strip_think

HERE = Path(__file__).parent

SAMPLE_FACT_SHEET = """QUESTION: evening run near Example Nagar tomorrow, user has asthma
PROFILE: asthma (US AQI limit 100)
BEST: Lakeside Garden [E1], Thu 22 Oct 17:00-18:00 [E2], score 81/100 [E3]
FACTS:
[E4] feels-like temperature 29.5 °C at 17:00
[E5] rain probability 10% at 17:00
[E6] US AQI 72 at 17:00 (forecast, ~40 km grid)
[E7] wind from W at 14 km/h at 17:00; exposure index 0.08 (modelled), main upwind source: none significant
[E8] travel 12 min by road (OSRM), ~19 min with evening traffic (heuristic)
[E9] sunset 18:05
[E10] at 08:00 the same park has exposure index 0.41 because wind blows from Ring Road (E)
CARRY: water about 0.6 L [E4]; reliever inhaler
WATCH FOR: sunset at 18:05 [E9]
AVOID: 08:00-10:00, traffic air drifting from Ring Road [E10]
OTHER OPTIONS: Hill View Park [E11] 17:00-18:00 score 74 [E12]
LIMITATIONS: air quality forecast is a ~40 km grid; travel times exclude live traffic"""


def score_intent(pred: Intent, gold: dict) -> dict:
    loc_ok = (gold["location"] is None and not pred.location_text) or (
        gold["location"] is not None and pred.location_text is not None
        and (gold["location"] in pred.location_text.lower() or pred.location_text.lower() in gold["location"]))
    return {
        "activity": pred.activity == gold["activity"] or (not gold["in_scope"] and not pred.in_scope),
        "location": loc_ok,
        "profiles": sorted(pred.profiles) == sorted(gold["profiles"]),
        "in_scope": pred.in_scope == gold["in_scope"],
        "time": bool(pred.time_text) == gold["has_time"],
    }


def run(model_path: str, do_compose: bool) -> dict:
    from llama_cpp import Llama, LlamaGrammar
    llm = Llama(model_path=model_path, n_ctx=4096, n_threads=os.cpu_count(), verbose=False)
    grammar = LlamaGrammar.from_json_schema(json.dumps(intent_json_schema()), verbose=False)
    gold = [json.loads(l) for l in (HERE / "intent_gold.jsonl").read_text().splitlines() if l.strip()]
    totals = {"activity": 0, "location": 0, "profiles": 0, "in_scope": 0, "time": 0}
    rows, lat = [], []
    for g in gold:
        t0 = time.monotonic()
        out = llm.create_completion(render_chatml(intent_messages(g["q"])), grammar=grammar, max_tokens=220,
                                    temperature=0.0, stop=["<|im_end|>"])
        lat.append(time.monotonic() - t0)
        try:
            pred = Intent(**json.loads(strip_think(out["choices"][0]["text"]))).normalised()
            s = score_intent(pred, g)
        except Exception as exc:  # noqa: BLE001
            pred, s = None, {k: False for k in totals}
            print("  parse error:", exc)
        for k in totals:
            totals[k] += int(s[k])
        rows.append({"q": g["q"], "pred": pred.model_dump() if pred else None, "score": s})
        if not all(s.values()):
            print("  MISS", {k: v for k, v in s.items() if not v}, "|", g["q"], "->",
                  pred and (pred.activity, pred.location_text, pred.profiles, pred.in_scope, pred.time_text))
    n = len(gold)
    result = {"model": os.path.basename(model_path), "n": n,
              "accuracy": {k: round(v / n, 3) for k, v in totals.items()},
              "intent_latency_s": {"mean": round(sum(lat) / n, 2), "max": round(max(lat), 2)}}
    if do_compose:
        t0 = time.monotonic()
        out = llm.create_completion(render_chatml(compose_messages(SAMPLE_FACT_SHEET)), max_tokens=380,
                                    temperature=0.2, stop=["<|im_end|>"])
        text = strip_think(out["choices"][0]["text"])
        sheet_numbers = set(re.findall(r"\d+(?:\.\d+)?", SAMPLE_FACT_SHEET))
        answer_numbers = re.findall(r"\d+(?:\.\d+)?", re.sub(r"\[E\d+\]", "", text))
        foreign = [x for x in answer_numbers if x not in sheet_numbers]
        result["compose"] = {"latency_s": round(time.monotonic() - t0, 1), "foreign_numbers": foreign,
                             "citations": len(re.findall(r"\[E\d+\]", text)), "text": text}
    return result


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    results = []
    for path in args:
        print(f"== {path}")
        r = run(path, "--compose" in sys.argv)
        print(json.dumps({k: v for k, v in r.items() if k != "compose"}, indent=1))
        if "compose" in r:
            print("compose:", {k: v for k, v in r["compose"].items() if k != "text"})
            print(r["compose"]["text"])
        results.append(r)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "bakeoff.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
