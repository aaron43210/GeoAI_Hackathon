#!/usr/bin/env bash
# Run notebook 01 (data) then 02 (LoRA fine-tune + publish to the -v2 repos) with no browser open, so the job keeps
# going after you close the laptop. Run it on the GPU server, inside tmux (or screen):
#
#   tmux new -s train                      # a session that survives disconnects
#   conda activate terrapulse              # the Python 3.11 env
#   cd ~/GeoAI_Hackathon                   # then switch it to the round-2 branch (once):
#   git fetch --depth 1 origin +round2-dgx:refs/remotes/origin/round2-dgx && git checkout -B round2-dgx origin/round2-dgx
#   read -s HF_TOKEN && export HF_TOKEN    # paste a Hugging Face WRITE token (not echoed, not saved in history)
#   bash training/run_round2.sh
#   # press Ctrl-b then d to detach, close the laptop; later: ssh in again and `tmux attach -t train`
#
# Without tmux:  nohup bash training/run_round2.sh > round2.out 2>&1 &     (follow with: tail -f round2.out)
# Executed notebooks, with all their output, are saved in training/runs/. Re-running resumes the data steps.
set -euo pipefail
cd "$(dirname "$0")/.."
# Some servers set PYTHONPATH to their own (older) Python packages, which leaks into the venv and makes the system
# Jupyter run the notebooks on the wrong Python. Clear it and use this environment's Python for everything.
unset PYTHONPATH
PY="$(command -v python)"
# ONLY=02 bash training/run_round2.sh   -> skip notebook 01 (data already built in training/data/)
NOTEBOOKS="01_build_data_colab 02_finetune_lora_colab"
[ "${ONLY:-}" = "02" ] && NOTEBOOKS="02_finetune_lora_colab"
: "${HF_TOKEN:?export HF_TOKEN first (a Hugging Face write token)}"

"$PY" - <<'PY'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Python {sys.version.split()[0]} ({sys.executable}) is too old for the training libraries: "
             "activate a Python 3.10+ environment first (e.g. conda activate terrapulse)")
print("Python", sys.version.split()[0], sys.executable)
PY
"$PY" -m pip -q install --upgrade pip
"$PY" -m pip -q install --ignore-installed nbconvert ipykernel huggingface_hub
# A kernel bound to exactly this Python, so nbconvert can't fall back to the server's default one.
"$PY" -m ipykernel install --user --name terrapulse-run --display-name "TerraPulse run ($("$PY" --version))" >/dev/null

# Check the token before doing anything: it must be valid and allowed to write.
"$PY" - <<'PY'
from huggingface_hub import whoami
try:
    me = whoami()
except Exception as exc:
    raise SystemExit(f"HF_TOKEN was rejected ({exc.__class__.__name__}). Create a new token with WRITE access at "
                     "https://huggingface.co/settings/tokens, then: read -s HF_TOKEN && export HF_TOKEN")
role = me.get("auth", {}).get("accessToken", {}).get("role")
if role == "read":
    raise SystemExit("HF_TOKEN is read-only: create a token with WRITE access")
print("Hugging Face user:", me["name"], "| token role:", role)
PY

# Lock round 1 first: tag the exact commits the live Space serves as "v1-stable" (no-op if the tag exists).
# Round 2 publishes only to the separate -v2 repos, so these are never overwritten.
"$PY" - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
V1 = {"aaronr43210/terrapulse-qwen3-1.7b": "58c3aeb1a32e9eb454199b5e951246cd4769d057",
      "aaronr43210/terrapulse-qwen3-1.7b-lora": "1d869bcae4e8d1f45da6e4851b512f1d0ecef2f7",
      "aaronr43210/terrapulse-qwen3-1.7b-GGUF": "ffaff4392895e8c96f4abc7e37529bd898b53abb"}
for repo, sha in V1.items():
    tags = {t.name for t in api.list_repo_refs(repo).tags}
    if "v1-stable" not in tags:
        api.create_tag(repo, tag="v1-stable", revision=sha, tag_message="Round-1 model served by the live Space")
    print("locked", repo, "v1-stable ->", sha[:8])
PY

mkdir -p training/runs
stamp=$(date +%Y%m%d-%H%M)
for nb in $NOTEBOOKS; do
  echo "$(date '+%F %T') start  $nb"
  "$PY" -m nbconvert --to notebook --execute "training/$nb.ipynb" \
    --output-dir training/runs --output "${nb}_${stamp}" \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=terrapulse-run
  echo "$(date '+%F %T') done   $nb  -> training/runs/${nb}_${stamp}.ipynb"
done
echo "Round 2 finished. Compare the BEFORE/AFTER table at the end of training/runs/02_finetune_lora_colab_${stamp}.ipynb"
