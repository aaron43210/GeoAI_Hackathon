"""Collect REAL fact sheets by running the live pipeline (SLM off) on generated questions. Runs in Colab.

    python training/build_compose_data.py --n 160 --out training/data/factsheets.jsonl

Each row keeps what the grounding verifier needs, so teacher drafts can be filtered with the exact runtime check.
Be gentle with the free APIs: requests are sequential and paced, and responses are cached by the app.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ["SLM_ENABLED"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from app.knowledge import get_knowledge  # noqa: E402
from app.main import app  # noqa: E402

sys.path.insert(0, str(ROOT / "training"))
from gen_intent_data import PLACES, PROFILE_PHRASES, TIMES  # noqa: E402

ASK = ["Where can I go for {act} {time} near {p}{prof}?", "Best time {time} for {act} around {p}{prof}",
       "Is {time} okay for {act} in {p}{prof}?", "{act} {time} near {p}, any good spots{prof}?"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--out", default=str(ROOT / "training" / "data" / "factsheets.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    kb = get_knowledge()
    acts = list(kb.activities.values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with TestClient(app) as client, open(out, "a", encoding="utf-8") as fh:
        while done < args.n:
            a = rng.choice(acts)
            prof = rng.choice(PROFILE_PHRASES)[0]
            q = rng.choice(ASK).format(act=rng.choice(a.synonyms), time=rng.choice(TIMES[:22]), p=rng.choice(PLACES),
                                       prof=(", " + prof) if prof else "")
            try:
                r = client.post("/api/v1/ask", json={"question": q, "overrides": {"accept_first_place": True}}).json()
            except Exception as exc:  # noqa: BLE001
                print("error", exc)
                continue
            if not r.get("fact_sheet"):
                print("skip", r.get("status"), q)
                continue
            rec = r.get("recommendation") or r.get("least_bad") or {}
            row = {"question": q, "status": r["status"], "fact_sheet": r["fact_sheet"],
                   "template_answer": r["answer_markdown"], "evidence_ids": [e["id"] for e in r["evidence"]],
                   "evidence_values": {e["id"]: e["value"] for e in r["evidence"]},
                   "known_names": r["heatmap"]["venues"] + [r["place"]["name"]],
                   "must_mention": (rec.get("venue") or {}).get("name"),
                   "recommended": r.get("recommendation") is not None}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            done += 1
            print(f"[{done}/{args.n}] {r['status']:16} {q}")
            time.sleep(args.pause)


if __name__ == "__main__":
    main()
