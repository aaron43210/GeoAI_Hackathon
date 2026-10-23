"""Publish accurate, repo-specific model cards for the three TerraPulse repos (adapter, merged, GGUF).

    HF_TOKEN=... python training/update_model_cards.py              # round 1 repos
    HF_TOKEN=... SUFFIX=-v2 python training/update_model_cards.py   # round 2 repos (-v2)

Evaluation numbers below are the ones we can defend: the clean intent gold set (excluded from training) and live
end-to-end runs on questions never seen in training. The notebook's in-training grounding score is reported with its
caveat (those 20 fact sheets were in the training data in the 21 Oct run).
"""
from __future__ import annotations

import os
import re

from huggingface_hub import HfApi

USER = "aaronr43210"
CODE = "https://github.com/aaron43210/GeoAI_Hackathon"
DATA = f"{USER}/terrapulse-sft-data"
BASE = "Qwen/Qwen3-1.7B"
SUFFIX = os.environ.get("SUFFIX", "")

REPOS = {
    "terrapulse-qwen3-1.7b": {
        "kind": "merged",
        "meta": "library_name: transformers\nbase_model_relation: finetune",
        "title": "merged weights (safetensors, `from_pretrained`)",
        "usage": f'''```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "{USER}/terrapulse-qwen3-1.7b"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto", device_map="auto")

# Prompts must match training: ChatML + an empty think block (thinking off). The runtime builds them with
# backend/app/slm/prompts.py and backend/app/slm/runtime.py:render_chatml in the code repo.
prompt = ("<|im_start|>system\\nYou extract structured fields ... Reply with JSON only.<|im_end|>\\n"
          "<|im_start|>user\\nevening jog near Indiranagar tomorrow?<|im_end|>\\n"
          "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n")
ids = tok(prompt, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(**ids, max_new_tokens=200, do_sample=False)[0][ids["input_ids"].shape[1]:],
                 skip_special_tokens=True))
```''',
        "files": "`model.safetensors`, tokenizer files, `chat_template.jinja`, plus `prompts.json`, `intent_schema.json` "
                 "and `terrapulse_config.json` (prompts, JSON schema and decoding settings used at runtime).",
    },
    "terrapulse-qwen3-1.7b-lora": {
        "kind": "adapter",
        "meta": "library_name: peft\nbase_model_relation: adapter",
        "title": "LoRA adapter (PEFT)",
        "usage": f'''```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("{BASE}", torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "{USER}/terrapulse-qwen3-1.7b-lora")
tok = AutoTokenizer.from_pretrained("{USER}/terrapulse-qwen3-1.7b-lora")
# or load the adapter straight away: AutoModelForCausalLM.from_pretrained("{USER}/terrapulse-qwen3-1.7b-lora")
```''',
        "files": "`adapter_model.safetensors`, `adapter_config.json` (base: `Qwen/Qwen3-1.7B`, r=16, α=32), tokenizer.",
    },
    "terrapulse-qwen3-1.7b-GGUF": {
        "kind": "gguf",
        "meta": "library_name: gguf\nbase_model_relation: quantized",
        "title": "GGUF quantisations for llama.cpp (what the live API serves)",
        "usage": f'''```python
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
path = hf_hub_download("{USER}/terrapulse-qwen3-1.7b-GGUF", "terrapulse-qwen3-1.7b.Q4_K_M.gguf")
llm = Llama(model_path=path, n_ctx=4096)
out = llm.create_completion("<|im_start|>user\\nkids cricket this weekend near Thane?<|im_end|>\\n"
                            "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n", max_tokens=200, temperature=0)
print(out["choices"][0]["text"])
```
The TerraPulse API loads this file by default (`SLM_REPO` / `SLM_FILE` in `backend/app/config.py`) and constrains the
intent output with a JSON-schema grammar.''',
        "files": "`terrapulse-qwen3-1.7b.Q4_K_M.gguf` (≈1.1 GB, default; ~2–4 s per intent on a laptop CPU) and "
                 "`terrapulse-qwen3-1.7b.Q8_0.gguf` (≈1.8 GB, higher fidelity).",
    },
}

BODY = """
# {repo}

**TerraPulse AI** small language model, {title}. A LoRA fine-tune of [`{base}`](https://huggingface.co/{base})
(Apache-2.0, 1.72 B parameters with tied input/output embeddings, under the hackathon's 3 B limit), built for the
**GeoAI Hackathon @ CSRE, IIT Bombay**: an assistant that recommends where and when to head outdoors.

| | |
|---|---|
| Code (API, pipeline, verifier, training notebooks) | {code} |
| Adapter / merged / GGUF | [`-lora`](https://huggingface.co/{user}/terrapulse-qwen3-1.7b-lora) · [merged](https://huggingface.co/{user}/terrapulse-qwen3-1.7b) · [`-GGUF`](https://huggingface.co/{user}/terrapulse-qwen3-1.7b-GGUF) |
| Training data | [`{data}`](https://huggingface.co/datasets/{data}) |
| Files in this repo | {files} |

## Usage
{usage}

## What the model does (and doesn't)
The model handles two **language** tasks inside a deterministic system:
1. **Intent extraction**: a free-form question (English, Hinglish, typos) → JSON with activity, place text, verbatim
   time phrase, radius, people profiles (asthma, senior, child, pregnancy, cardiac, heat-sensitive…), preferences and scope.
2. **Grounded answer writing**: a fact sheet of **live** data (Open-Meteo weather, CAMS air quality, OpenStreetMap
   venues/roads/industry/green/water, OSRM travel times) → a short recommendation that cites fact ids such as `[E4]`.

It holds **no** weather or place knowledge of its own. Dates, scores and safety limits are computed by code, and every
answer is checked by a deterministic verifier (numbers + units, times, place names, citation ids and values, severity
wording). An answer that fails is repaired once or replaced by a data template.

## Why this base model
Measured bake-off on CPU (Q4_K_M, 24 hand-written questions): **Qwen3-1.7B** scored activity 0.917, location 1.0,
profiles 0.958, scope 0.917 at 2.2 s per intent; Qwen3.5-2B scored 0.875 / 0.958 / 0.958 / 0.958 at 4.9 s.
Excluded on size or licence: Phi-3/Phi-4-mini (3.8 B), Ministral-3-3B (3.85 B), Granite-4.0-micro (3.4 B),
Llama-3.2-3B (3.21 B, non-OSI licence), Qwen2.5-3B (research licence), SmolLM3 (3.08 B), Gemma-3n-E2B (5.4 B).

## Training
- LoRA r=16, α=32, dropout 0.05 on all attention and MLP projections; 2 epochs, lr 2e-4 cosine, completion-only loss;
  Unsloth + TRL on a free Google Colab T4.
- 3,002 training examples:
  - **intent**: 3,000 generated questions with exact labels (Indian localities, 13 activities, 7 profiles, Hinglish,
    typos, out-of-scope requests, follow-ups); the evaluation questions were excluded;
  - **answers**: 160 fact sheets from the live pipeline; 144 answers drafted by the open `Qwen/Qwen3-4B` and kept only
    because they passed the verifier, plus 16 template answers.
- No proprietary or hosted LLM was used anywhere (data, training or runtime).
- Prompt format: ChatML with an empty `<think></think>` block, identical in training and inference.

## Evaluation
**Intent extraction**, 24 hand-written questions not in the training data (Q4_K_M, CPU):

| metric | base Qwen3-1.7B | TerraPulse fine-tune |
|---|---|---|
| activity | 0.917 | **1.0** |
| location | 1.0 | 0.958 |
| people profiles | 0.958 | **1.0** |
| in/out of scope | 0.917 | **0.958** |
| valid JSON | 1.0 (grammar-constrained) | 1.0 |

**End to end, live data**, questions never seen in training (full API with verifier):

| test | base | fine-tune |
|---|---|---|
| answer drafts passing the verifier first time (20-question set) | 4 / 13 | **12 / 14** |
| correct status (answer / clarify / decline / not available), 20 questions | 18 / 20 | **19 / 20** |
| correct status, 24 further scenarios | — | **23 / 24** (the miss was a parser bug, since fixed) |
| activity and profile extraction (44 questions) | 100 % | **100 %** |

> The training notebook also reported a grounding pass rate of 1.0 vs 0.55. Those 20 fact sheets were part of the
> training data in that run, so treat that figure as optimistic; the live first-draft numbers above are the clean ones.

## Intended use
Research and demonstration inside an outdoor-activity assistant that supplies live data. Health-related limits are
conservative heuristics based on US EPA AQI guidance and heat-index bands; this is not medical advice.

## Limitations
- Wording can be terse. Factual safety relies on the external verifier, not on the model alone.
- Trained mostly on Indian place names, English and Hinglish.
- Only 2 of the 160 training fact sheets contained road/industry exposure facts (OpenStreetMap's Overpass servers
  were unavailable during data collection), so those phrasings are under-represented.

## Licence
Apache-2.0, the same as the base model. Map data © OpenStreetMap contributors; weather data by Open-Meteo.com (CC BY 4.0).
"""


BODY_V2 = BODY[:BODY.index("## Training")] + """## Training (round 2)
- Same LoRA recipe as round 1: r=16, α=32, dropout 0.05 on all attention and MLP projections; 2 epochs, lr 2e-4
  cosine, completion-only loss; Unsloth + TRL, this time on one 80 GB NVIDIA GPU of a college DGX server.
- Data ([`{data}`](https://huggingface.co/datasets/{data})), about 4,000 examples before a ~5 % validation split:
  - **intent**: 3,000 generated questions with exact labels (same generator as round 1; evaluation questions excluded);
  - **explanations**: 500 fact sheets from the live pipeline. The open teacher `Qwen/Qwen3-32B` (bf16, thinking off,
    run on our own GPUs at training time only) wrote short friendly explanations in the app's current chat format;
    246 passed the runtime verifier and were kept, plus 125 code-written explanations;
  - **follow-ups**: 714 answers to "why?"-style questions ("is the air okay for this?", "why is it not recommended?")
    from the same fact sheets, each kept only because it passed the verifier.
  - The last 20 fact sheets were held out of training for evaluation.
- No proprietary or hosted LLM was used anywhere (data, training or runtime).
- Prompt format: ChatML with an empty `<think></think>` block, identical in training and inference.

## Evaluation (23 Oct 2026, round 1 vs round 2, same code and prompts)
**Intent extraction**, 24 hand-written questions not in the training data (Q4_K_M, laptop CPU):

| metric | round 1 | round 2 (this model) |
|---|---|---|
| activity | 1.0 | 0.958 |
| location | 0.958 | **1.0** |
| people profiles | 1.0 | 1.0 |
| in/out of scope | 0.958 | 0.917 |
| time phrase | 0.958 | 0.958 |
| seconds per intent | 2.50 | **2.36** |

In the app a rule parser is merged with the model output, so the one scope miss here ("indoor gym" read as yoga) is
still declined end to end.

**End to end, live data**, 20 questions never seen in training (full API with the verifier, current chat prompt):

| test | round 1 | round 2 |
|---|---|---|
| model-written explanations passing the verifier | 4 / 14 | **10 / 14** (+1 after one repair) |
| correct status (answer / clarify / decline / not available) | 19 / 20 | 19 / 20 |
| activity and profile extraction | 100 % | 100 % |
| median seconds per question (laptop CPU) | 14.9 | 15.7 |

Round 1 was trained on an older, structured answer format, which is why its drafts fail more often with the app's
current chat prompt. The one status miss in both rounds is an ambiguous place name, where asking is the right outcome.

""" + BODY[BODY.index("## Intended use"):]


def main() -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))    # falls back to the saved `hf auth login` token
    for name, spec in REPOS.items():
        repo_id = f"{USER}/{name}{SUFFIX}"
        header = (f"---\nlicense: apache-2.0\nbase_model: {BASE}\n{spec['meta']}\ndatasets:\n- {DATA}\n"
                  "language:\n- en\n- hi\npipeline_tag: text-generation\n"
                  "tags:\n- geoai\n- outdoor-activity\n- grounded-generation\n- qwen3\n- small-language-model\n"
                  + ("- lora\n" if spec["kind"] != "gguf" else "- gguf\n- llama-cpp\n") + "---\n")
        body = BODY_V2 if SUFFIX == "-v2" else BODY
        if SUFFIX:   # point every repo link and snippet at this round's repos, e.g. ...-lora -> ...-lora-v2
            body = re.sub(r"terrapulse-qwen3-1\.7b(-lora|-GGUF)?(?=[`)\"'\]])", lambda m: m.group(0) + SUFFIX, body)
            usage = re.sub(r"terrapulse-qwen3-1\.7b(-lora|-GGUF)?(?=[\"'])", lambda m: m.group(0) + SUFFIX, spec["usage"])
            files = spec["files"].replace(", plus `prompts.json`, `intent_schema.json` "
                                          "and `terrapulse_config.json` (prompts, JSON schema and decoding settings used at runtime)", "")
        else:
            usage, files = spec["usage"], spec["files"]
        card = header + body.format(repo=repo_id, title=spec["title"], base=BASE, code=CODE, user=USER,
                                    data=DATA, files=files, usage=usage)
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo_id,
                        commit_message=f"Accurate model card for {repo_id}: usage, clean evaluation, links")
        print("updated", f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
