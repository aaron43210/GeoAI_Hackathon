"""Model evaluation on identical inputs: base Qwen3-1.7B vs TerraPulse round 1 vs round 2 (llama.cpp, CPU, Q4_K_M).

    python -m eval.model_eval --model base=/p/base.gguf --model v1=/p/v1.gguf --model v2=/p/v2.gguf \
        --data /path/to/terrapulse-sft-data   # folder with factsheets.jsonl and compose.jsonl from the HF dataset

Measures, per model:
  * intent extraction (84 hand-labelled questions: eval/intent_gold.jsonl + eval/intent_test.jsonl, none in training):
    accuracy, precision, recall, macro-F1 for activity, location, people profiles and in/out of scope;
  * decision (routing) accuracy: plan / ask a clarifying question / decline, from the model alone and from the full
    system (model merged with the rule parser, exactly as app/agent/understand.py does);
  * robustness: the same questions with typos, chat shorthand and filler noise;
  * fairness: counterfactual place swaps (same question, places in different regions) and subgroup gaps
    (English vs Hinglish, with vs without a health profile);
  * generation on the 20 held-out live fact sheets (never in training): groundedness (runtime verifier pass rate),
    hallucination rate, claim-level faithfulness, decision consistency, ROUGE and BERTScore against the verified
    answers of the Qwen3-32B teacher, for both the explanation and "why?" follow-ups;
  * model inference latency p50 / p95 and tokens per second for each task.
Writes eval/results/model_eval.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from app.agent import rule_parser
from app.agent.understand import merge_intents
from app.evidence.grounding import verify
from app.schemas.intent import Intent, intent_json_schema
from app.slm.prompts import chat_messages, explain_messages, intent_messages
from app.slm.runtime import cpu_budget, render_chatml, strip_think

HERE = Path(__file__).parent
PROFILES = ["asthma", "senior", "child", "pregnancy", "cardiac", "heat_sensitive", "mobility_limited"]

# ---------------------------------------------------------------- test material (evaluation only)
SHORTHAND = [("tomorrow", "tmrw"), ("morning", "mrng"), ("evening", "evng"), ("near", "nr"), ("weekend", "wknd"),
             ("please", "pls"), ("with", "wid"), ("before", "b4"), ("for", "4")]
FAIRNESS_TEMPLATES = [
    ("evening jog near {p} tomorrow", "running", []),
    ("I have asthma, morning walk around {p} this weekend?", "walking", ["asthma"]),
    ("kids cricket near {p} on saturday afternoon", "cricket", ["child"]),
    ("cycling in {p} day after tomorrow morning", "cycling", []),
    ("picnic with my parents near {p} this sunday", "picnic", ["senior"]),
    ("yoga in a park near {p} tomorrow at 6 am", "yoga", []),
]
FAIRNESS_PLACES = {"north": ["Lajpat Nagar", "Gomti Nagar"], "west": ["Navrangpura", "Kothrud"],
                   "south": ["T Nagar", "Jayanagar 4th Block"], "east": ["Salt Lake", "Ganeshguri"]}


def perturb_typo(q: str) -> str:
    words = q.split()
    order = sorted(range(len(words)), key=lambda i: -len(words[i]))[:2]
    for i in order:
        w = words[i]
        if len(w) > 4 and w[0].islower():
            words[i] = w[:2] + w[3:]                      # drop the third letter of the two longest lowercase words
    return " ".join(words)


def perturb_shorthand(q: str) -> str:
    s = re.sub(r"[?!.,]", "", q.lower())
    for a, b in SHORTHAND:
        s = re.sub(rf"\b{a}\b", b, s)
    return s


def perturb_noise(q: str) -> str:
    return f"hey quick question!! {q} thanks a lot"


# ---------------------------------------------------------------- metrics helpers
def pct(xs) -> Optional[float]:
    xs = list(xs)
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def multiclass(gold: List[str], pred: List[str]) -> Dict[str, float]:
    labels = sorted(set(gold) | set(pred))
    ps, rs, fs = [], [], []
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        p_, r_, f_ = prf(tp, fp, fn)
        ps.append(p_); rs.append(r_); fs.append(f_)
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    return {"accuracy": round(100 * acc, 1), "precision": round(100 * statistics.mean(ps), 1),
            "recall": round(100 * statistics.mean(rs), 1), "macro_f1": round(100 * statistics.mean(fs), 1)}


def gap(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return round(abs(a - b), 1) if a is not None and b is not None else None


def pctl(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 2)


def loc_ok(gold: Optional[str], pred: Optional[str]) -> bool:
    pred = (pred or "").lower().strip()
    if gold is None:
        return not pred
    return bool(pred) and (gold in pred or pred in gold)


def route(intent: Intent) -> str:
    if not intent.in_scope:
        return "decline"
    if not intent.activity or not (intent.location_text or "").strip():
        return "clarify"                                  # "near me" also needs a location the eval doesn't supply
    return "plan"


def gold_route(g: dict) -> str:
    if not g["in_scope"]:
        return "decline"
    return "plan" if g["activity"] and g["location"] else "clarify"


def joint_ok(g: dict, p: Intent) -> bool:
    if not g["in_scope"]:
        return not p.in_scope
    return (p.in_scope and p.activity == g["activity"] and loc_ok(g["location"], p.location_text)
            and sorted(p.profiles) == sorted(g["profiles"]))


HALLUCINATION = ("not in fact sheet", "unknown citation", "place/name", "cited next to")
DECISION = ("not mentioned", "doesn't say so", "used for a not-recommended", "health warning", "but the data flags",
            "but the data does not flag", "calls ", "lists other options")


def clean_text(s: str) -> str:
    s = re.sub(r"\[E\d+\]", "", s)
    s = re.sub(r"[*_#>`]", "", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- model wrapper
class Model:
    def __init__(self, path: str, gpu_layers: int = 0):
        from llama_cpp import Llama, LlamaGrammar
        self.llm = Llama(model_path=path, n_ctx=4096, n_threads=cpu_budget(), n_gpu_layers=gpu_layers, verbose=False,
                         seed=0)
        self.grammar = LlamaGrammar.from_json_schema(json.dumps(intent_json_schema()), verbose=False)

    def gen(self, messages, max_tokens: int, temperature: float, grammar=None):
        t0 = time.perf_counter()
        out = self.llm.create_completion(render_chatml(messages), max_tokens=max_tokens, temperature=temperature,
                                         top_p=0.9, repeat_penalty=1.15, grammar=grammar, seed=0)
        dt = time.perf_counter() - t0
        return strip_think(out["choices"][0]["text"]), dt, out["usage"]["completion_tokens"]

    def intent(self, q: str):
        text, dt, toks = self.gen(intent_messages(q), 260, 0.0, self.grammar)
        try:
            return Intent.from_model_output(json.loads(text)).normalised(), dt, toks, True
        except Exception:  # noqa: BLE001 - counted as a JSON failure, scored as an empty intent
            return Intent(in_scope=True), dt, toks, False


# ---------------------------------------------------------------- evaluation
def evaluate(name: str, path: str, tests: List[dict], robust: List[dict], fair: List[dict],
             sheets: List[dict], explains: List[dict], refs: Dict[int, str], gpu_layers: int = 0) -> dict:
    m = Model(path, gpu_layers)
    lat = defaultdict(list)
    toks = defaultdict(list)

    def run_intents(items):
        out = []
        for it in items:
            p, dt, n, ok = m.intent(it["q"])
            lat["intent"].append(dt); toks["intent"].append(n / dt if dt else 0)
            sysp = merge_intents(it["q"], rule_parser.parse(it["q"]), p)
            out.append((it, p, sysp, ok))
        return out

    print(f"[{name}] intent: {len(tests)} test + {len(robust)} perturbed + {len(fair)} counterfactual", flush=True)
    clean = run_intents(tests)
    pert = run_intents(robust)
    cf = run_intents(fair)

    res: Dict[str, object] = {"model": name, "file": os.path.basename(path)}
    for view, idx in (("model", 1), ("system", 2)):
        rows = [(g, r[idx]) for r in clean for g in [r[0]]]
        inscope = [(g, p) for g, p in rows if g["in_scope"]]
        act = multiclass([str(g["activity"]) for g, _ in inscope], [str(p.activity) for _, p in inscope])
        tp = sum(g["location"] is not None and loc_ok(g["location"], p.location_text) for g, p in inscope)
        fp = sum(bool((p.location_text or "").strip()) and not loc_ok(g["location"], p.location_text) for g, p in inscope)
        fn = sum(g["location"] is not None and not loc_ok(g["location"], p.location_text) for g, p in inscope)
        lp, lr, lf = prf(tp, fp, fn)
        loc = {"accuracy": pct(loc_ok(g["location"], p.location_text) for g, p in inscope),
               "precision": round(100 * lp, 1), "recall": round(100 * lr, 1), "macro_f1": round(100 * lf, 1)}
        ps, rs, fs = [], [], []
        for prof in PROFILES:
            t_ = sum(prof in g["profiles"] and prof in p.profiles for g, p in rows)
            f_ = sum(prof not in g["profiles"] and prof in p.profiles for g, p in rows)
            n_ = sum(prof in g["profiles"] and prof not in p.profiles for g, p in rows)
            a, b, c = prf(t_, f_, n_)
            ps.append(a); rs.append(b); fs.append(c)
        prof_m = {"accuracy": pct(sorted(g["profiles"]) == sorted(p.profiles) for g, p in rows),
                  "precision": round(100 * statistics.mean(ps), 1), "recall": round(100 * statistics.mean(rs), 1),
                  "macro_f1": round(100 * statistics.mean(fs), 1)}
        scope = multiclass(["in" if g["in_scope"] else "out" for g, _ in rows],
                           ["in" if p.in_scope else "out" for _, p in rows])
        o_tp = sum(not g["in_scope"] and not p.in_scope for g, p in rows)
        o_fp = sum(g["in_scope"] and not p.in_scope for g, p in rows)
        o_fn = sum(not g["in_scope"] and p.in_scope for g, p in rows)
        scope["out_of_scope_precision"], scope["out_of_scope_recall"], _ = [round(100 * x, 1) for x in prf(o_tp, o_fp, o_fn)]
        dec = [gold_route(g) == route(p) for g, p in rows]
        res[f"intent_{view}"] = {"n": len(rows), "activity": act, "location": loc, "profiles": prof_m, "in_scope": scope,
                                 "joint_accuracy": pct(joint_ok(g, p) for g, p in rows),
                                 "decision_accuracy": pct(dec), "decision_ci95": wilson(sum(dec), len(dec)),
                                 "decision_by_class": {k: pct(gold_route(g) == route(p) for g, p in rows if gold_route(g) == k)
                                                       for k in ("plan", "clarify", "decline")}}
    res["valid_json"] = pct(r[3] for r in clean + pert + cf)

    # robustness: perturbed questions must still be read correctly (model alone and full system)
    res["robustness"] = {
        "n": len(pert),
        "model_success": pct(joint_ok(r[0], r[1]) for r in pert),
        "system_success": pct(joint_ok(r[0], r[2]) for r in pert),
        "model_clean_same_items": pct(joint_ok(c[0], c[1]) for c in clean if any(c[0]["q"] == r[0]["src"] for r in pert)),
        "by_kind": {k: pct(joint_ok(r[0], r[1]) for r in pert if r[0]["kind"] == k) for k in ("typo", "shorthand", "noise")},
    }

    # fairness: counterfactual place swaps + subgroup gaps
    by_t = defaultdict(list)
    for it, p, _, _ in cf:
        by_t[it["template"]].append((it, p))
    consistent = [len({(p.activity, tuple(sorted(p.profiles)), p.in_scope) for _, p in v}) == 1 for v in by_t.values()]
    region = {r: pct(joint_ok(it, p) for it, p in sum(by_t.values(), []) if it["region"] == r) for r in FAIRNESS_PLACES}
    groups = {
        "english": [joint_ok(g, p) for g, p, _, _ in clean if g.get("lang", "en") == "en" and g["in_scope"]],
        "hinglish": [joint_ok(g, p) for g, p, _, _ in clean if g.get("lang") == "hi-en"],
        "with_profile": [joint_ok(g, p) for g, p, _, _ in clean if g["profiles"]],
        "without_profile": [joint_ok(g, p) for g, p, _, _ in clean if g["in_scope"] and not g["profiles"]],
    }
    g_acc = {k: pct(v) for k, v in groups.items()}
    res["fairness"] = {
        "counterfactual_consistency": pct(consistent), "templates": len(consistent), "per_region_accuracy": region,
        "region_gap": gap(max(v for v in region.values() if v is not None), min(v for v in region.values() if v is not None)),
        "subgroup_accuracy": g_acc, "subgroup_n": {k: len(v) for k, v in groups.items()},
        "language_gap": gap(g_acc["english"], g_acc["hinglish"]),
        "profile_gap": gap(g_acc["with_profile"], g_acc["without_profile"]),
    }

    # generation: explanation (chat prompt) on held-out sheets, then "why?" follow-ups
    gen = {"explain": [], "followup": []}
    print(f"[{name}] generation: {len(sheets)} explanations + {len(explains)} follow-ups", flush=True)
    for i, s in enumerate(sheets):
        a, dt, n = m.gen(chat_messages(s["fact_sheet"]), 520, 0.2)
        lat["explanation"].append(dt); toks["explanation"].append(n / dt if dt else 0)
        r = verify(a, s["fact_sheet"], set(s["evidence_ids"]), s["known_names"], s["must_mention"], s["recommended"],
                   s.get("evidence_values"))
        gen["explain"].append({"sheet": i, "answer": a, "violations": r.violations, "claims": r.facts_checked,
                               "ref": refs.get(i)})
    for e in explains:
        s = sheets[e["sheet"]]
        a, dt, n = m.gen(explain_messages(s["fact_sheet"], e["question"]), 260, 0.2)
        lat["follow-up"].append(dt); toks["follow-up"].append(n / dt if dt else 0)
        r = verify(a, s["fact_sheet"], set(s["evidence_ids"]), s["known_names"], None, True, s.get("evidence_values"))
        gen["followup"].append({"sheet": e["sheet"], "question": e["question"], "answer": a,
                                "violations": r.violations, "claims": r.facts_checked, "ref": e["ref"]})

    for kind, rows in gen.items():
        hall = [sum(any(h in v for h in HALLUCINATION) for v in x["violations"]) for x in rows]
        claims = sum(x["claims"] for x in rows)
        res[f"gen_{kind}"] = {
            "n": len(rows),
            "groundedness": pct(not x["violations"] for x in rows),
            "groundedness_ci95": wilson(sum(not x["violations"] for x in rows), len(rows)),
            "hallucination_rate": pct(h > 0 for h in hall),
            "faithfulness_claims": round(100 * (1 - sum(hall) / claims), 1) if claims else None,
            "claims_checked": claims,
            "decision_consistency": pct(not any(d in v for v in x["violations"] for d in DECISION) for x in rows),
            "empty_answers": sum(not x["answer"].strip() for x in rows),
        }
    res["latency_s"] = {k: {"p50": pctl(v, .5), "p95": pctl(v, .95), "n": len(v)} for k, v in lat.items()}
    res["tokens_per_s"] = {k: round(statistics.median(v), 1) for k, v in toks.items() if v}
    res["_generations"] = gen
    del m
    return res


def add_reference_scores(results: List[dict]) -> None:
    """ROUGE-1/ROUGE-L F1 and BERTScore F1 (roberta-large, baseline-rescaled) against the teacher's verified answers."""
    from bert_score import score as bscore
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    for res in results:
        for kind in ("explain", "followup"):
            pairs = [(clean_text(x["answer"]), clean_text(x["ref"])) for x in res["_generations"][kind] if x.get("ref")]
            if not pairs:
                continue
            cands, refs = [c or "-" for c, _ in pairs], [r for _, r in pairs]
            r1 = [rs.score(r, c)["rouge1"].fmeasure for c, r in zip(cands, refs)]
            rl = [rs.score(r, c)["rougeL"].fmeasure for c, r in zip(cands, refs)]
            _, _, f = bscore(cands, refs, lang="en", rescale_with_baseline=True, verbose=False)
            res[f"gen_{kind}"].update({"ref_pairs": len(pairs), "rouge1_f1": round(100 * statistics.mean(r1), 1),
                                       "rougeL_f1": round(100 * statistics.mean(rl), 1),
                                       "bertscore_f1": round(100 * float(f.mean()), 1)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="name=/path/to/model.gguf")
    ap.add_argument("--data", required=True, help="folder with factsheets.jsonl and compose.jsonl")
    ap.add_argument("--heldout", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only the first N items of each set")
    ap.add_argument("--gpu-layers", type=int, default=0, help="-1 = all layers on GPU (needs a CUDA/Metal build)")
    ap.add_argument("--rescore-only", action="store_true", help="only (re)compute ROUGE/BERTScore from saved results")
    args = ap.parse_args()

    tests = [json.loads(l) for f in ("intent_gold.jsonl", "intent_test.jsonl") for l in open(HERE / f) if l.strip()]
    robust = []
    for g in tests:
        if g["in_scope"] and g["activity"] and g["location"] and g.get("lang", "en") == "en":
            for kind, fn in (("typo", perturb_typo), ("shorthand", perturb_shorthand), ("noise", perturb_noise)):
                q = fn(g["q"])
                if q != g["q"]:
                    robust.append({**g, "q": q, "src": g["q"], "kind": kind})
    fair = [{"q": t.format(p=p), "activity": a, "location": p.lower(), "profiles": prof, "in_scope": True,
             "template": t, "region": region}
            for t, a, prof in FAIRNESS_TEMPLATES for region, ps in FAIRNESS_PLACES.items() for p in ps]

    sheets_all = [json.loads(l) for l in open(Path(args.data) / "factsheets.jsonl")]
    sheets = sheets_all[-args.heldout:]                    # held out of training by notebook 02
    refs, explains = {}, []
    for c in (json.loads(l) for l in open(Path(args.data) / "compose.jsonl")):
        if c["source"] != "teacher":
            continue
        user = c["messages"][-2]["content"]
        for i, s in enumerate(sheets):
            if s["fact_sheet"] in user:
                if c["task"] == "compose":
                    refs.setdefault(i, c["messages"][-1]["content"])
                else:
                    q = user.split("FOLLOW-UP QUESTION:", 1)[-1].strip()
                    explains.append({"sheet": i, "question": q, "ref": c["messages"][-1]["content"]})

    if args.limit:
        n = args.limit
        tests, robust, fair, explains = tests[:n], robust[:n], fair[:2 * n], [e for e in explains if e["sheet"] < 2][:2]
        sheets = sheets[:2]
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    path = out / "model_eval.json"
    meta = {"machine": f"{platform.machine()} · {os.cpu_count()} cores · {platform.platform(terse=True)}",
            "runtime": (f"llama.cpp (llama-cpp-python), {'GPU' if args.gpu_layers else 'CPU only'}, Q4_K_M, "
                        "temperature 0 for intent / 0.2 for text, seed 0"),
            "intent_questions": len(tests), "perturbed": len(robust), "counterfactual": len(fair),
            "heldout_sheets": len(sheets), "followups": len(explains), "explanation_refs": len(refs)}
    if args.rescore_only:
        results = json.loads(path.read_text())["results"]
    else:
        results = []
        for spec in args.model:
            name, mpath = spec.split("=", 1)
            results.append(evaluate(name, mpath, tests, robust, fair, sheets, explains, refs, args.gpu_layers))
            path.write_text(json.dumps({"meta": meta, "results": results}, indent=1, ensure_ascii=False))
    try:
        add_reference_scores(results)
    except Exception as exc:  # noqa: BLE001 - keep the model results; rerun with --rescore-only
        print("reference scores skipped:", exc)
    path.write_text(json.dumps({"meta": meta, "results": results}, indent=1, ensure_ascii=False))
    print(json.dumps({"meta": meta, "results": [{k: v for k, v in r.items() if k != "_generations"} for r in results]},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
