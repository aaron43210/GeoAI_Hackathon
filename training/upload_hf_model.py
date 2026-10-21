"""Publish the TerraPulse model repo on the Hugging Face Hub (run in Colab, or anywhere with HF_TOKEN set).

Mode "base" (before fine-tuning): satisfies the rule "if you use a base model unchanged, still create a repository with
your configuration and a model card naming the base model". The repo contains the prompts, generation config,
intent JSON schema and a model card; `from_pretrained` loads the base weights via the documented id.

    HF_TOKEN=... python training/upload_hf_model.py --mode base

Mode "finetuned" is done by notebook 02 (adapter, merged weights, GGUF and card are pushed there).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

BASE = "Qwen/Qwen3-1.7B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="aaronr43210")
    ap.add_argument("--repo", default="terrapulse-qwen3-1.7b")
    ap.add_argument("--mode", choices=["base"], default="base")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    from app.schemas.intent import intent_json_schema
    from app.slm import prompts

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repo_id = f"{args.user}/{args.repo}"
    api.create_repo(repo_id, exist_ok=True)
    card = (ROOT / "training" / "MODEL_CARD.md").read_text()
    card = card.replace("{{REPO}}", repo_id).replace("{{N_TRAIN}}", "0 (base model used unchanged)")
    card = card.replace("{{EVAL_TABLE}}", "Base-model bake-off on this machine (Q4_K_M, CPU, 24 gold questions): activity 0.917, "
                        "location 1.0, profiles 0.958, scope 0.917; 0 invented numbers in the composition test. "
                        "Fine-tuned results will replace this table.")
    card += (f"\n\n## Status: base model, configuration only\nThis repository currently documents how TerraPulse uses "
             f"`{BASE}` **unchanged** (prompts, decoding settings, schema). Load the weights with:\n\n"
             f"```python\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n"
             f"tok = AutoTokenizer.from_pretrained('{BASE}')\nmodel = AutoModelForCausalLM.from_pretrained('{BASE}')\n```\n")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "README.md").write_text(card)
        (p / "terrapulse_config.json").write_text(json.dumps({
            "base_model": BASE, "serving": {"runtime": "llama.cpp", "gguf_repo": "unsloth/Qwen3-1.7B-GGUF",
                                            "gguf_file": "Qwen3-1.7B-Q4_K_M.gguf"},
            "chat_format": "ChatML with empty <think></think> (thinking disabled)",
            "intent": {"temperature": 0.0, "max_tokens": 260, "grammar": "JSON schema (intent_schema.json)"},
            "compose": {"temperature": 0.2, "max_tokens": 380, "top_p": 0.9},
        }, indent=1))
        (p / "intent_schema.json").write_text(json.dumps(intent_json_schema(), indent=1))
        (p / "prompts.json").write_text(json.dumps({
            "intent_system": prompts.intent_system(), "few_shot": prompts.FEW_SHOT,
            "compose_system": prompts.COMPOSE_SYSTEM, "structure_ok": prompts.STRUCTURE_OK,
            "structure_not_recommended": prompts.STRUCTURE_NOT_RECOMMENDED, "repair": prompts.REPAIR_NOTE,
        }, indent=1, ensure_ascii=False))
        api.upload_folder(folder_path=d, repo_id=repo_id, commit_message="TerraPulse configuration + model card (base)")
    print(f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
