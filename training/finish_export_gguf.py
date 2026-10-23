#!/usr/bin/env python3
"""Export the fine-tuned merged model to GGUF (Q4_K_M and Q8_0) and publish to Hugging Face.

Runs standalone on the GPU/DGX server after training completes.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

HF_USER = "aaronr43210"
SUFFIX = "-v2"
GGUF_REPO = f"{HF_USER}/terrapulse-qwen3-1.7b-GGUF{SUFFIX}"
LORA_REPO = f"{HF_USER}/terrapulse-qwen3-1.7b-lora{SUFFIX}"
MERGED_REPO = f"{HF_USER}/terrapulse-qwen3-1.7b{SUFFIX}"

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    sys.exit("Error: HF_TOKEN environment variable not set. Export your Hugging Face write token.")

api = HfApi(token=TOKEN)

# Locate merged model weights
possible_paths = [
    Path(f"/jupyter/sods.capstone01/GeoAI_Hackathon/{MERGED_REPO}"),
    Path(MERGED_REPO),
    Path.home() / "GeoAI_Hackathon" / MERGED_REPO,
]
merged_dir = None
for p in possible_paths:
    if p.exists() and (p / "config.json").exists():
        merged_dir = p
        break

if not merged_dir:
    sys.exit(f"Could not find merged model directory. Checked: {[str(p) for p in possible_paths]}")

print(f"[1/5] Located merged weights at: {merged_dir}")

# Ensure gguf python package is installed
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf"], check=True)

# Clone or find llama.cpp
llama_dir = Path.home() / "llama.cpp"
if not llama_dir.exists():
    print("[2/5] Cloning llama.cpp...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(llama_dir)], check=True)
else:
    print(f"[2/5] Using existing llama.cpp at {llama_dir}")

# Build llama-quantize binary
quantize_bin = None
candidates = [
    llama_dir / "build" / "bin" / "llama-quantize",
    llama_dir / "build" / "llama-quantize",
    llama_dir / "llama-quantize",
]
for c in candidates:
    if c.exists() and os.access(c, os.X_OK):
        quantize_bin = c
        break

if not quantize_bin:
    print("[3/5] Building llama-quantize (CPU only, no extra deps)...")
    build_dir = llama_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_cmd = ["cmake", "-B", str(build_dir), "-S", str(llama_dir), "-DGGML_CURL=OFF", "-DGGML_CUDA=OFF"]
    res = subprocess.run(cmake_cmd)
    if res.returncode == 0:
        subprocess.run(["cmake", "--build", str(build_dir), "-j", "--target", "llama-quantize"], check=True)
    else:
        # Fallback to make
        subprocess.run(["make", "-C", str(llama_dir), "llama-quantize", "-j4"], check=True)

    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            quantize_bin = c
            break

if not quantize_bin or not quantize_bin.exists():
    sys.exit("Error: Failed to find or build llama-quantize binary.")

print(f"[3/5] Quantizer binary ready: {quantize_bin}")

# Convert HF to GGUF BF16
out_dir = Path("gguf_out")
out_dir.mkdir(parents=True, exist_ok=True)
bf16_gguf = out_dir / "terrapulse-qwen3-1.7b.BF16.gguf"

converter_script = llama_dir / "convert_hf_to_gguf.py"
print(f"[4/5] Converting HF safetensors to GGUF BF16...")
subprocess.run([
    sys.executable, str(converter_script),
    str(merged_dir),
    "--outfile", str(bf16_gguf),
    "--outtype", "bf16",
], check=True)

# Quantize to Q4_K_M and Q8_0
q4_gguf = out_dir / "terrapulse-qwen3-1.7b.Q4_K_M.gguf"
q8_gguf = out_dir / "terrapulse-qwen3-1.7b.Q8_0.gguf"

print("Quantizing to Q4_K_M...")
subprocess.run([str(quantize_bin), str(bf16_gguf), str(q4_gguf), "Q4_K_M"], check=True)

print("Quantizing to Q8_0...")
subprocess.run([str(quantize_bin), str(bf16_gguf), str(q8_gguf), "Q8_0"], check=True)

# Clean up large intermediate bf16 gguf
if bf16_gguf.exists():
    bf16_gguf.unlink()

# Upload to Hugging Face
print(f"[5/5] Uploading GGUF files to {GGUF_REPO}...")
api.create_repo(GGUF_REPO, exist_ok=True)
api.upload_file(
    path_or_fileobj=str(q4_gguf),
    path_in_repo="terrapulse-qwen3-1.7b.Q4_K_M.gguf",
    repo_id=GGUF_REPO,
)
print("Uploaded terrapulse-qwen3-1.7b.Q4_K_M.gguf")

api.upload_file(
    path_or_fileobj=str(q8_gguf),
    path_in_repo="terrapulse-qwen3-1.7b.Q8_0.gguf",
    repo_id=GGUF_REPO,
)
print("Uploaded terrapulse-qwen3-1.7b.Q8_0.gguf")

print("\nSUCCESS! All round-2 models (adapter, merged, GGUF) are published to Hugging Face:")
print(f"  LoRA adapter : https://huggingface.co/{LORA_REPO}")
print(f"  Merged model : https://huggingface.co/{MERGED_REPO}")
print(f"  GGUF models  : https://huggingface.co/{GGUF_REPO}")
