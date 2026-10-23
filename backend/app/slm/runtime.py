"""Local small-language-model runtime (≤3B, open weights) via llama.cpp, with an optional transformers backend.

Prompts are rendered as ChatML (used by Qwen3/Qwen3.5 and by our fine-tune) with an empty <think> block, which is
how Qwen's own chat template disables thinking. JSON outputs are grammar-constrained from a JSON schema, so the
model cannot emit invalid JSON.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

THINK_OFF = "<think>\n\n</think>\n\n"


def cpu_budget() -> int:
    """CPUs this process may actually use. Containers often report the host's core count, and oversubscribing
    llama.cpp threads on a 2-CPU quota makes inference crawl, so read the cgroup quota first."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 2


def render_chatml(messages: List[Dict[str, str]]) -> str:
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(parts) + "<|im_start|>assistant\n" + THINK_OFF


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


@dataclass
class ModelInfo:
    backend: str = "none"
    repo: Optional[str] = None
    file: Optional[str] = None
    path_used: str = "none"            # finetuned | base | none
    load_ms: int = 0
    error: Optional[str] = None
    calls: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class SLMRuntime:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.info = ModelInfo(backend=self.settings.SLM_BACKEND if self.settings.SLM_ENABLED else "none")
        self._llm = None
        self._tok = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._llm is not None

    # ------------------------------------------------------------------ loading
    async def load(self) -> None:
        if not self.settings.SLM_ENABLED or self.settings.SLM_BACKEND == "none":
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        started = time.monotonic()
        try:
            if self.settings.SLM_BACKEND == "transformers":
                self._load_transformers()
            else:
                self._load_llama_cpp()
            self.info.load_ms = round((time.monotonic() - started) * 1000)
            logger.info("SLM ready: %s %s (%s) in %sms", self.info.repo, self.info.file, self.info.path_used,
                        self.info.load_ms)
        except Exception as exc:  # noqa: BLE001 - runtime must start even without a model
            self.info.error = f"{type(exc).__name__}: {exc}"
            logger.error("SLM failed to load, rule/template fallbacks will be used: %s", self.info.error)

    def _resolve_gguf(self) -> tuple[str, str, str, str]:
        s = self.settings
        if s.SLM_LOCAL_PATH:
            return s.SLM_LOCAL_PATH, "local", os.path.basename(s.SLM_LOCAL_PATH), "local"
        from huggingface_hub import hf_hub_download
        try:
            return hf_hub_download(s.SLM_REPO, s.SLM_FILE), s.SLM_REPO, s.SLM_FILE, "finetuned"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fine-tuned GGUF %s/%s unavailable (%s); using base model", s.SLM_REPO, s.SLM_FILE, exc)
            return hf_hub_download(s.SLM_BASE_REPO, s.SLM_BASE_FILE), s.SLM_BASE_REPO, s.SLM_BASE_FILE, "base"

    def _load_llama_cpp(self) -> None:
        from llama_cpp import Llama
        path, repo, fname, which = self._resolve_gguf()
        threads = self.settings.SLM_THREADS or cpu_budget()
        self.info.extra["threads"] = threads
        self._llm = Llama(model_path=path, n_ctx=self.settings.SLM_CTX, n_threads=threads, verbose=False,
                          n_gpu_layers=self.settings.SLM_GPU_LAYERS,
                          use_mmap=self.settings.SLM_USE_MMAP)
        self.info.repo, self.info.file, self.info.path_used = repo, fname, which

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        repo = self.settings.SLM_HF_MODEL
        self._tok = AutoTokenizer.from_pretrained(repo)
        cuda = torch.cuda.is_available()
        self._llm = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16 if cuda else torch.float32)
        if cuda:
            self._llm = self._llm.to("cuda")
        self.info.repo, self.info.file, self.info.path_used = repo, None, "finetuned"

    # ------------------------------------------------------------------ inference
    def complete(self, messages: List[Dict[str, str]], *, json_schema: Optional[dict] = None,
                 max_tokens: int = 256, temperature: Optional[float] = None) -> str:
        if not self.ready:
            raise RuntimeError("SLM not loaded")
        temp = self.settings.SLM_TEMPERATURE if temperature is None else temperature
        prompt = render_chatml(messages)
        with self._lock:
            self.info.calls += 1
            if self.settings.SLM_BACKEND == "transformers":
                return strip_think(self._hf_generate(prompt, max_tokens, temp))
            kwargs: Dict[str, Any] = {"max_tokens": max_tokens, "temperature": temp, "top_p": self.settings.SLM_TOP_P,
                                      "repeat_penalty": self.settings.SLM_REPETITION_PENALTY,
                                      "stop": ["<|im_end|>", "<|endoftext|>"]}
            if json_schema is not None:
                from llama_cpp import LlamaGrammar
                kwargs["grammar"] = LlamaGrammar.from_json_schema(json.dumps(json_schema), verbose=False)
            out = self._llm.create_completion(prompt, **kwargs)
            return strip_think(out["choices"][0]["text"])

    def stream(self, messages: List[Dict[str, str]], *, max_tokens: Optional[int] = None,
               temperature: Optional[float] = None) -> Iterator[str]:
        if not self.ready:
            raise RuntimeError("SLM not loaded")
        max_tokens = max_tokens or self.settings.SLM_MAX_TOKENS_ANSWER
        if self.settings.SLM_BACKEND == "transformers":
            yield self.complete(messages, max_tokens=max_tokens, temperature=temperature)
            return
        temp = self.settings.SLM_TEMPERATURE if temperature is None else temperature
        with self._lock:
            self.info.calls += 1
            for chunk in self._llm.create_completion(render_chatml(messages), max_tokens=max_tokens, temperature=temp,
                                                     top_p=self.settings.SLM_TOP_P, repeat_penalty=self.settings.SLM_REPETITION_PENALTY,
                                                     stop=["<|im_end|>", "<|endoftext|>"], stream=True):
                yield chunk["choices"][0]["text"]

    def _hf_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        inputs = self._tok(prompt, return_tensors="pt").to(self._llm.device)
        out = self._llm.generate(**inputs, max_new_tokens=max_tokens, do_sample=temperature > 0,
                                 temperature=max(temperature, 1e-3),
                                 repetition_penalty=self.settings.SLM_REPETITION_PENALTY)
        return self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.info.backend, "repo": self.info.repo, "file": self.info.file,
                "path_used": self.info.path_used if self.ready else "none", "loaded": self.ready,
                "load_ms": self.info.load_ms, "error": self.info.error, "threads": self.info.extra.get("threads")}
