---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
language: [en, hi]
tags: [geoai, outdoor-activity, grounded-generation, lora, qwen3, small-language-model]
pipeline_tag: text-generation
---

# {{REPO}}

**TerraPulse AI** small language model: a LoRA fine-tune of [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B)
(Apache-2.0, ≈2.0 B parameters including embeddings; under the 3 B limit) for the GeoAI Hackathon @ CSRE, IIT Bombay.

Related repos: `aaronr43210/terrapulse-qwen3-1.7b-lora` (adapter), `aaronr43210/terrapulse-qwen3-1.7b` (merged,
`from_pretrained`), `aaronr43210/terrapulse-qwen3-1.7b-GGUF` (Q4_K_M / Q8_0 for llama.cpp CPU serving),
dataset `aaronr43210/terrapulse-sft-data`.

## What it does (and what it does not)
The model does two **language** jobs inside a larger, deterministic system:
1. **Intent extraction**: free-form (incl. Hinglish, typos) question → JSON: activity, place text, time phrase
   (verbatim), radius, profiles (asthma, senior, child, pregnancy, cardiac, heat-sensitive…), preferences, scope.
2. **Grounded answer writing**: a fact sheet of live data (Open-Meteo weather and CAMS air quality, OpenStreetMap
   venues/roads/industry/green/water, OSRM travel) → a short recommendation citing fact ids like `[E4]`.

It does **not** know weather or places. All facts come from live APIs at query time, and every answer is checked by a
deterministic grounding verifier (numbers + units, times, place names, citations, severity wording). Failing answers
are repaired once or replaced by a template.

## Training
- Method: LoRA (r=16, α=32, dropout 0.05, all attention + MLP projections), 2 epochs, lr 2e-4 cosine,
  completion-only loss, Unsloth + TRL on a free Colab T4.
- Data ({{N_TRAIN}} training examples):
  - intent: programmatically generated questions with exact labels (Indian localities, 13 activities, profiles,
    Hinglish, typos, out-of-scope, follow-ups);
  - compose: fact sheets produced by the live pipeline; target answers drafted by the **open** model
    `Qwen/Qwen3-4B` and kept **only if they pass the grounding verifier**, plus a minority of template answers.
  - No proprietary/hosted LLM was used to create data. The hackathon's example questions and our eval gold set are excluded.
- Prompt format: ChatML with an empty `<think></think>` block (thinking disabled), identical at train and inference.

## Evaluation (held-out gold set of 24 hand-written questions + 20 held-out live fact sheets)
{{EVAL_TABLE}}

## Intended use
Research and demo use in an outdoor-activity assistant that supplies live data. Not a medical device: health-related
limits are conservative heuristics based on US EPA AQI guidance and heat-index bands.

## Limitations
Small model phrasing can be terse; the system relies on the verifier for factual safety. Trained mostly on Indian
place names and English/Hinglish.

## Licence
Apache-2.0 (same as the base model).
