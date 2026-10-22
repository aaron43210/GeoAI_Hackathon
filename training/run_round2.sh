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
: "${HF_TOKEN:?export HF_TOKEN first (a Hugging Face write token)}"

python -m pip -q install nbconvert ipykernel huggingface_hub

# Lock round 1 first: tag the exact commits the live Space serves as "v1-stable" (no-op if the tag exists).
# Round 2 publishes only to the separate -v2 repos, so these are never overwritten.
python - <<'PY'
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
for nb in 01_build_data_colab 02_finetune_lora_colab; do
  echo "$(date '+%F %T') start  $nb"
  jupyter nbconvert --to notebook --execute "training/$nb.ipynb" \
    --output-dir training/runs --output "${nb}_${stamp}" \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3
  echo "$(date '+%F %T') done   $nb  -> training/runs/${nb}_${stamp}.ipynb"
done
echo "Round 2 finished. Compare the BEFORE/AFTER table at the end of training/runs/02_finetune_lora_colab_${stamp}.ipynb"
