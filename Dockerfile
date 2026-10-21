# Single container: React UI + FastAPI API + local ≤3B SLM (llama.cpp, CPU). One URL for judges.
FROM node:20-alpine AS ui
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake && rm -rf /var/lib/apt/lists/*
WORKDIR /app/backend
COPY backend/requirements*.txt ./
# prebuilt CPU wheel for llama-cpp-python (falls back to source build)
RUN pip install --no-cache-dir -r requirements-core.txt && \
    pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
COPY backend/ ./
COPY --from=ui /ui/dist /app/frontend/dist
# Fetch the GGUF at build time so cold starts are fast (fine-tuned repo first, base model otherwise).
ARG SLM_REPO=aaronr43210/terrapulse-qwen3-1.7b-GGUF
ARG SLM_FILE=terrapulse-qwen3-1.7b.Q4_K_M.gguf
RUN python -c "from huggingface_hub import hf_hub_download as d;\
import sys;\
exec('try:\n print(d(\"$SLM_REPO\",\"$SLM_FILE\"))\nexcept Exception as e:\n print(\"fine-tuned GGUF not published yet:\", e); print(d(\"unsloth/Qwen3-1.7B-GGUF\",\"Qwen3-1.7B-Q4_K_M.gguf\"))')"
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
