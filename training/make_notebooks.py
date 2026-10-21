"""Builds the two Colab notebooks (run locally once: python training/make_notebooks.py). Training runs ONLY in Colab."""
import json
from pathlib import Path

HERE = Path(__file__).parent


def nb(cells):
    return {"cells": [{"cell_type": t, "metadata": {}, "source": s.strip("\n").splitlines(keepends=True),
                       **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells],
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


SETUP = r'''
# Get the project into the Colab runtime. Option A: clone your GitHub repo. Option B: upload GEOAI.zip.
REPO_URL = "https://github.com/aaron43210/GeoAI_Hackathon.git"   # set "" to upload GEOAI.zip instead
import os, subprocess, sys
if REPO_URL and os.path.exists("/content/GEOAI/.git"):
    subprocess.run(["git", "-C", "/content/GEOAI", "pull", "--ff-only"], check=True)
elif REPO_URL:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/GEOAI"], check=True)
elif not os.path.exists("/content/GEOAI"):
    from google.colab import files
    up = files.upload()                         # choose GEOAI.zip (project folder zipped, without node_modules)
    subprocess.run(["unzip", "-q", next(iter(up)), "-d", "/content/"], check=True)
%cd /content/GEOAI
!pip -q install -r backend/requirements-core.txt
sys.path.insert(0, "/content/GEOAI/backend")
'''

HF_LOGIN = r'''
# Hugging Face login: uses the Colab secret HF_TOKEN if available, otherwise asks (input is hidden, never saved).
import os
from huggingface_hub import login
token = os.environ.get("HF_TOKEN")
if not token:
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except Exception:
        token = None
if not token:
    from getpass import getpass
    token = getpass("Hugging Face write token: ")
login(token)
os.environ["HF_TOKEN"] = token
HF_USER = "aaronr43210"
'''

DATA_NB = [
    ("markdown", r'''
# TerraPulse AI · 01: build the fine-tuning data (Colab)
Open this notebook in VS Code with the **Colab extension** and pick a **T4 GPU** runtime. Nothing here runs on your laptop.

1. intent data: generated with exact labels (`training/gen_intent_data.py`)
2. real fact sheets: the live pipeline (SLM off) on ~160 generated questions (`training/build_compose_data.py`)
3. answer drafts: an **open** teacher (Qwen3-4B, fp16, batched) writes answers from each fact sheet; only drafts that pass
   the runtime **grounding verifier** are kept. No hosted/proprietary LLM is used anywhere.
'''),
    ("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    ("code", SETUP),
    ("code", "!python training/gen_intent_data.py --n 3000 --out training/data/intent.jsonl\n!head -c 600 training/data/intent.jsonl"),
    ("markdown", "### Real fact sheets from live APIs (≈15–25 min, paced to respect free-API limits)"),
    ("code", "!python training/build_compose_data.py --n 160 --pause 2 --out training/data/factsheets.jsonl"),
    ("markdown", "### Open teacher drafts → keep only answers that pass the grounding verifier"),
    ("code", r'''
!pip -q install -U "transformers>=4.51" accelerate
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.slm.prompts import compose_messages
from app.evidence.grounding import verify

# Open teacher in fp16 with batched generation: ~10-15 min for 160 sheets on a T4.
# (Qwen3-8B in 4-bit was ~9 min per sheet on a T4, far too slow.)
TEACHER = "Qwen/Qwen3-4B"          # Apache-2.0 open weights
tok = AutoTokenizer.from_pretrained(TEACHER, padding_side="left")
teacher = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=torch.float16, device_map="cuda")

def draft_batch(sheets, temperature):
    texts = [tok.apply_chat_template(compose_messages(s), tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for s in sheets]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = teacher.generate(**enc, max_new_tokens=320, do_sample=True, temperature=temperature,
                               top_p=0.9, pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]
'''),
    ("code", r'''
import os
rows = [json.loads(l) for l in open("training/data/factsheets.jsonl")]
kept, stats = [], {"teacher": 0, "template": 0, "dropped": 0}
if os.path.exists("training/data/compose.jsonl"):           # resume after an interruption
    kept = [json.loads(l) for l in open("training/data/compose.jsonl")]
    for k in kept: stats[k["source"]] += 1
done = {k["messages"][1]["content"] for k in kept}
todo = [r for r in rows if r["fact_sheet"] not in done]
print(len(kept), "kept so far,", len(todo), "to go")

def ok(r, a):
    return verify(a, r["fact_sheet"], set(r["evidence_ids"]), r["known_names"], r["must_mention"],
                  r["recommended"], r.get("evidence_values")).status == "passed"

def save(r, src, ans):
    msgs = compose_messages(r["fact_sheet"]) + [{"role": "assistant", "content": ans}]
    item = {"task": "compose", "source": src, "messages": msgs}
    kept.append(item); stats[src] += 1
    with open("training/data/compose.jsonl", "a") as fh: fh.write(json.dumps(item, ensure_ascii=False) + "\n")

B = 8                                   # lower to 4 if CUDA runs out of memory
for i in range(0, len(todo), B):
    batch = todo[i:i + B]
    retry = []
    for r, a in zip(batch, draft_batch([r["fact_sheet"] for r in batch], 0.5)):
        save(r, "teacher", a) if ok(r, a) else retry.append(r)
    if retry:
        for r, a in zip(retry, draft_batch([r["fact_sheet"] for r in retry], 0.8)):
            if ok(r, a): save(r, "teacher", a)
            elif stats["template"] < 0.25 * len(rows): save(r, "template", r["template_answer"])
            else: stats["dropped"] += 1
    print(f"{min(i + B, len(todo))}/{len(todo)}", stats)
print("final", stats, "kept", len(kept))
'''),
    ("code", HF_LOGIN),
    ("code", r'''
# Optional: publish the datasets (open licence) so the model card can link them
from huggingface_hub import HfApi
api = HfApi(); repo = f"{HF_USER}/terrapulse-sft-data"
api.create_repo(repo, repo_type="dataset", exist_ok=True)
for f in ["intent.jsonl", "compose.jsonl", "factsheets.jsonl"]:
    api.upload_file(path_or_fileobj=f"training/data/{f}", path_in_repo=f, repo_id=repo, repo_type="dataset")
print("https://huggingface.co/datasets/" + repo)
'''),
]

TRAIN_NB = [
    ("markdown", r'''
# TerraPulse AI · 02: LoRA fine-tune Qwen3-1.7B (Colab T4)
Open in VS Code via the **Colab extension**, runtime **T4 GPU**. Needs `training/data/intent.jsonl` and
`training/data/compose.jsonl` from notebook 01 (re-run 01 in this runtime, or download them from the HF dataset).

Base: `Qwen/Qwen3-1.7B` (Apache-2.0, ~2.0B params incl. embeddings, under the 3B limit). LoRA r=16, α=32, 2 epochs, completion-only loss.
Outputs pushed to the Hub: adapter, merged safetensors (`from_pretrained` works), GGUF Q4_K_M + Q8_0 (what the API serves).
'''),
    ("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    ("code", SETUP),
    ("code", '!pip -q install unsloth "trl>=0.20" datasets\n'),
    ("code", HF_LOGIN),
    ("code", r'''
import os, json, random
if not os.path.exists("training/data/compose.jsonl"):
    from huggingface_hub import hf_hub_download
    os.makedirs("training/data", exist_ok=True)
    for f in ["intent.jsonl", "compose.jsonl"]:
        p = hf_hub_download(f"{HF_USER}/terrapulse-sft-data", f, repo_type="dataset"); os.system(f"cp {p} training/data/{f}")
from app.slm.runtime import render_chatml
rows = [json.loads(l) for f in ["intent.jsonl", "compose.jsonl"] for l in open(f"training/data/{f}")]
random.Random(0).shuffle(rows)
def to_pc(r):   # prompt = exactly what the runtime sends (ChatML + empty <think>), completion = assistant text
    return {"prompt": render_chatml(r["messages"][:-1]), "completion": r["messages"][-1]["content"] + "<|im_end|>",
            "task": r["task"]}
data = [to_pc(r) for r in rows]
n_val = max(40, len(data) // 20)
val, train = data[:n_val], data[n_val:]
print(len(train), "train /", len(val), "val;", sum(d["task"] == "compose" for d in train), "compose")
'''),
    ("code", r'''
from unsloth import FastLanguageModel
import torch
BASE = "Qwen/Qwen3-1.7B"
model, tokenizer = FastLanguageModel.from_pretrained(BASE, max_seq_length=2048, load_in_4bit=False, dtype=None)
'''),
    ("markdown", "### Baseline eval (before fine-tuning): intent gold set + grounding pass rate"),
    ("code", r'''
from app.schemas.intent import Intent
from app.slm.prompts import intent_messages
from app.evidence.grounding import verify
import re, time
gold = [json.loads(l) for l in open("backend/eval/intent_gold.jsonl")]
sheets = [json.loads(l) for l in open("training/data/factsheets.jsonl")][-20:] if os.path.exists("training/data/factsheets.jsonl") else []

def gen(prompt, max_new=300):
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

def evaluate(tag):
    FastLanguageModel.for_inference(model)
    acc = {"activity": 0, "location": 0, "profiles": 0, "in_scope": 0, "json_ok": 0}
    for g in gold:
        txt = gen(render_chatml(intent_messages(g["q"])), 260)
        try:
            m = re.search(r"\{.*\}", txt, re.S); p = Intent(**json.loads(m.group(0))).normalised(); acc["json_ok"] += 1
        except Exception:
            continue
        acc["activity"] += p.activity == g["activity"] or (not g["in_scope"] and not p.in_scope)
        loc = (p.location_text or "").lower()
        acc["location"] += (g["location"] is None and not loc) or (g["location"] is not None and bool(loc) and (g["location"] in loc or loc in g["location"]))
        acc["profiles"] += sorted(p.profiles) == sorted(g["profiles"])
        acc["in_scope"] += p.in_scope == g["in_scope"]
    res = {k: round(v / len(gold), 3) for k, v in acc.items()}
    passed = 0
    from app.slm.prompts import compose_messages
    for s in sheets:
        a = gen(render_chatml(compose_messages(s["fact_sheet"])), 420)
        passed += verify(a, s["fact_sheet"], set(s["evidence_ids"]), s["known_names"], s["must_mention"], s["recommended"], s.get("evidence_values")).status == "passed"
    res["grounding_pass"] = round(passed / max(1, len(sheets)), 3)
    print(tag, res); return res
before = evaluate("BEFORE")
'''),
    ("code", r'''
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=0)
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(model=model, processing_class=tokenizer,
    train_dataset=Dataset.from_list([{k: d[k] for k in ("prompt", "completion")} for d in train]),
    eval_dataset=Dataset.from_list([{k: d[k] for k in ("prompt", "completion")} for d in val]),
    args=SFTConfig(output_dir="outputs", num_train_epochs=2, per_device_train_batch_size=4, gradient_accumulation_steps=4,
                   learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05, logging_steps=10,
                   eval_strategy="steps", eval_steps=100, save_strategy="no", max_length=2048,
                   completion_only_loss=True, fp16=not torch.cuda.is_bf16_supported(),
                   bf16=torch.cuda.is_bf16_supported(), report_to="none", seed=0))
stats = trainer.train(); print(stats)
'''),
    ("code", "after = evaluate(\"AFTER\")\nprint(json.dumps({\"before\": before, \"after\": after}, indent=1))"),
    ("markdown", "### Publish: adapter, merged model (`from_pretrained`), GGUF for CPU serving, model card"),
    ("code", r'''
from huggingface_hub import HfApi
api = HfApi()
LORA, MERGED, GGUF = f"{HF_USER}/terrapulse-qwen3-1.7b-lora", f"{HF_USER}/terrapulse-qwen3-1.7b", f"{HF_USER}/terrapulse-qwen3-1.7b-GGUF"
for r in (LORA, MERGED, GGUF): api.create_repo(r, exist_ok=True)
model.push_to_hub(LORA); tokenizer.push_to_hub(LORA)
model.push_to_hub_merged(MERGED, tokenizer, save_method="merged_16bit")
model.save_pretrained_gguf("gguf_out", tokenizer, quantization_method=["q4_k_m", "q8_0"])
import glob
for f in glob.glob("gguf_out*/*.gguf") + glob.glob("gguf_out/*.gguf"):
    q = "Q4_K_M" if "q4_k_m" in f.lower() else ("Q8_0" if "q8_0" in f.lower() else None)
    if q: api.upload_file(path_or_fileobj=f, path_in_repo=f"terrapulse-qwen3-1.7b.{q}.gguf", repo_id=GGUF)
print("uploaded")
'''),
    ("code", r'''
card = open("training/MODEL_CARD.md").read()
table = "| metric | base Qwen3-1.7B | TerraPulse LoRA |\n|---|---|---|\n" + "\n".join(
    f"| {k} | {before[k]} | {after[k]} |" for k in before)
card = card.replace("{{EVAL_TABLE}}", table).replace("{{N_TRAIN}}", str(len(train)))
for r in (LORA, MERGED, GGUF):
    api.upload_file(path_or_fileobj=card.replace("{{REPO}}", r).encode(), path_in_repo="README.md", repo_id=r)
open("training/eval_results.json", "w").write(json.dumps({"before": before, "after": after}, indent=1))
print(table)
'''),
    ("code", r'''
# Judges' check: the merged model loads with from_pretrained
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(MERGED, torch_dtype="auto", device_map="auto"); t = AutoTokenizer.from_pretrained(MERGED)
print(sum(p.numel() for p in m.parameters()) / 1e9, "B params, loaded OK")
'''),
]

if __name__ == "__main__":
    (HERE / "01_build_data_colab.ipynb").write_text(json.dumps(nb(DATA_NB), indent=1))
    (HERE / "02_finetune_lora_colab.ipynb").write_text(json.dumps(nb(TRAIN_NB), indent=1))
    print("notebooks written")
