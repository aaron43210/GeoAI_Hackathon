"""Live end-to-end evaluation against a running server (hits real free APIs; paced).

    python -m eval.run_eval --base http://localhost:8000 [--limit 10]

Reports status correctness, activity/profile extraction, grounding outcome (SLM passed / repaired / template),
whether any answer contains an uncited fact, and latency. Writes eval/results/live_eval.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--file", default="queries.jsonl")
    args = ap.parse_args()
    rows = [json.loads(l) for l in (HERE / args.file).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    out, lat = [], []
    with httpx.Client(timeout=180) as c:
        for r in rows:
            t0 = time.monotonic()
            try:
                res = c.post(f"{args.base}/api/v1/ask", json={"question": r["q"]}).json()
            except Exception as exc:  # noqa: BLE001
                out.append({"q": r["q"], "error": str(exc)})
                continue
            dt = time.monotonic() - t0
            lat.append(dt)
            intent = res.get("intent", {})
            item = {
                "q": r["q"], "status": res.get("status"), "status_ok": res.get("status") in r["expect_status"],
                "activity_ok": r.get("activity") is None or intent.get("activity") == r["activity"],
                "profiles_ok": set(r.get("profiles", [])) <= set(intent.get("profiles", [])),
                "grounding": (res.get("grounding") or {}).get("status"), "path": res.get("answer_path"),
                "seconds": round(dt, 1), "venue": ((res.get("recommendation") or res.get("least_bad") or {})
                                                   .get("venue") or {}).get("name"),
            }
            out.append(item)
            print(f"{'✔' if item['status_ok'] and item['activity_ok'] and item['profiles_ok'] else '✘'} "
                  f"{str(item['status']):20} {str(item['grounding'] or '-'):10} {item['seconds']:5}s  {r['q']}")
            time.sleep(args.pause)
    ok = [o for o in out if "error" not in o]
    summary = {
        "n": len(out), "errors": len(out) - len(ok),
        "status_acc": round(sum(o["status_ok"] for o in ok) / max(1, len(ok)), 3),
        "activity_acc": round(sum(o["activity_ok"] for o in ok) / max(1, len(ok)), 3),
        "profile_recall": round(sum(o["profiles_ok"] for o in ok) / max(1, len(ok)), 3),
        "slm_answer_rate": round(sum(1 for o in ok if (o["path"] or "").startswith("slm")) /
                                 max(1, sum(1 for o in ok if o["path"])), 3),
        "latency_p50_s": round(statistics.median(lat), 1) if lat else None,
        "latency_max_s": round(max(lat), 1) if lat else None,
    }
    print(json.dumps(summary, indent=1))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "live_eval.json").write_text(json.dumps({"summary": summary, "rows": out}, indent=1,
                                                                ensure_ascii=False))


if __name__ == "__main__":
    main()
