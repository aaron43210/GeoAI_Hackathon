"""Clean latency benchmark: one model at a time, nothing else running, same inputs for every model.

    python -m eval.latency_bench --gpu-layers -1 --data <sft-data folder> --model base=a.gguf --model v1=b.gguf ...

Times 40 intent extractions, the 20 held-out explanations and 20 follow-ups per model and writes the p50 / p95 into
eval/results/<--into> (replacing latency_s and tokens_per_s), so the quality evaluation can run in parallel without
distorting latency.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from app.slm.prompts import chat_messages, explain_messages
from eval.model_eval import HERE, Model, pctl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--into", default="model_eval.json")
    args = ap.parse_args()
    tests = [json.loads(l) for f in ("intent_gold.jsonl", "intent_test.jsonl") for l in open(HERE / f) if l.strip()][:40]
    sheets = [json.loads(l) for l in open(Path(args.data) / "factsheets.jsonl")][-20:]
    path = HERE / "results" / args.into
    doc = json.loads(path.read_text())
    by = {r["model"]: r for r in doc["results"]}
    for spec in args.model:
        name, mpath = spec.split("=", 1)
        m = Model(mpath, args.gpu_layers)
        m.gen(chat_messages(sheets[0]["fact_sheet"]), 16, 0.2)          # warm-up, not timed
        lat, tps = defaultdict(list), defaultdict(list)
        for t in tests:
            _, dt, n, _ = m.intent(t["q"])
            lat["intent"].append(dt); tps["intent"].append(n / dt)
        for s in sheets:
            _, dt, n = m.gen(chat_messages(s["fact_sheet"]), 520, 0.2)
            lat["explanation"].append(dt); tps["explanation"].append(n / dt if dt else 0)
            _, dt, n = m.gen(explain_messages(s["fact_sheet"], "why this place and time?"), 260, 0.2)
            lat["follow-up"].append(dt); tps["follow-up"].append(n / dt if dt else 0)
        if name in by:
            by[name]["latency_s"] = {k: {"p50": pctl(v, .5), "p95": pctl(v, .95), "n": len(v)} for k, v in lat.items()}
            by[name]["tokens_per_s"] = {k: round(statistics.median(v), 1) for k, v in tps.items()}
        print(name, {k: (pctl(v, .5), pctl(v, .95)) for k, v in lat.items()}, flush=True)
        del m
    doc["meta"]["latency_note"] = "latency measured separately, one model at a time with nothing else running"
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
