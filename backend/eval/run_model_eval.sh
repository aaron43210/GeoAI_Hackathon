#!/usr/bin/env bash
# Base Qwen3-1.7B vs TerraPulse v1 vs v2 on identical inputs (see eval/model_eval.py), e.g. on a GPU server:
#
#   tmux new -s eval
#   source ~/tp311/bin/activate                    # any Python 3.10+ env
#   cd ~/GeoAI_Hackathon && git pull               # main
#   GPU=1 bash backend/eval/run_model_eval.sh      # GPU=1 builds llama.cpp with CUDA and offloads all layers
#
# Downloads (public, no token needed): the three Q4_K_M GGUFs and the round-2 dataset (held-out fact sheets).
# Results: backend/eval/results/model_eval.json. Latency numbers describe the machine it runs on.
set -eo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
PY="$(command -v python)"
WORK="${EVAL_WORK:-$PWD/eval/.cache}"          # inside the repo (git-ignored); some servers leave $HOME unset
export HF_HOME="${HF_HOME:-$WORK/hf}"
mkdir -p "$WORK/models" "$WORK/data"

if [ "${GPU:-0}" = "1" ]; then
  CMAKE_ARGS="-DGGML_CUDA=on" "$PY" -m pip install -q --no-cache-dir --force-reinstall --no-deps llama-cpp-python
  "$PY" -m pip install -q typing-extensions numpy diskcache jinja2
  LAYERS=-1
else
  "$PY" -m pip install -q llama-cpp-python
  LAYERS=0
fi
"$PY" -m pip install -q -r requirements-core.txt rouge-score bert-score

"$PY" - "$WORK" <<'PY'
import sys
from huggingface_hub import hf_hub_download
work = sys.argv[1]
for repo, fname, out in [("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf", "base.gguf"),
                         ("aaronr43210/terrapulse-qwen3-1.7b-GGUF", "terrapulse-qwen3-1.7b.Q4_K_M.gguf", "v1.gguf"),
                         ("aaronr43210/terrapulse-qwen3-1.7b-GGUF-v2", "terrapulse-qwen3-1.7b.Q4_K_M.gguf", "v2.gguf")]:
    p = hf_hub_download(repo, fname, local_dir=f"{work}/models/{out}.d")
    print("model", out, p)
for f in ("factsheets.jsonl", "compose.jsonl"):
    print("data", hf_hub_download("aaronr43210/terrapulse-sft-data", f, repo_type="dataset", local_dir=f"{work}/data"))
PY

M="$WORK/models"
"$PY" -m eval.model_eval --gpu-layers "$LAYERS" --data "$WORK/data" \
  --model base="$M/base.gguf.d/Qwen3-1.7B-Q4_K_M.gguf" \
  --model v1="$M/v1.gguf.d/terrapulse-qwen3-1.7b.Q4_K_M.gguf" \
  --model v2="$M/v2.gguf.d/terrapulse-qwen3-1.7b.Q4_K_M.gguf" 2>&1 | grep -v "^llama\|ggml\|^load"
echo "results: $(pwd)/eval/results/model_eval.json"
