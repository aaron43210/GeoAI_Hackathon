"""TerraPulse AI on a Hugging Face ZeroGPU Space.

ZeroGPU containers have no usable CPU budget for model inference, so here the fine-tuned model runs on the GPU:
  * the merged model (aaronr43210/terrapulse-qwen3-1.7b) is loaded with transformers at import time and moved to
    "cuda" BEFORE ZeroGPU's startup report, which is what lets ZeroGPU pack the weights;
  * every generation runs inside a @spaces.GPU function (executed by ZeroGPU in a forked worker, so only plain
    strings/numbers cross the boundary);
  * each HTTP request is exposed to ZeroGPU as a Gradio request, so the caller's X-IP-Token is used for quota.
If ZeroGPU is unavailable or out of quota, the pipeline falls back to its rule parser and data template, so the
API keeps answering with grounded results.
"""
import os

os.environ.setdefault("SLM_BACKEND", "transformers")

import spaces  # must be imported before torch / anything CUDA-related on ZeroGPU

import gradio as gr
import uvicorn
from gradio.context import LocalContext

from app.main import app as fastapi_app
from app.slm.runtime import SLMRuntime

# ---------------------------------------------------------------- model on the (virtual) GPU, loaded at import
_slm = SLMRuntime()
_slm._load_sync()
fastapi_app.state.preloaded_slm = _slm
_local_generate = SLMRuntime._hf_generate


@spaces.GPU(duration=45)
def _gpu_generate(prompt: str, max_tokens: int, temperature: float) -> str:
    return _local_generate(_slm, prompt, max_tokens, temperature)


def _generate_on_gpu(self, prompt: str, max_tokens: int, temperature: float) -> str:
    return _gpu_generate(prompt, max_tokens, temperature)


SLMRuntime._hf_generate = _generate_on_gpu


@fastapi_app.middleware("http")
async def _zerogpu_request_context(request, call_next):
    """Let ZeroGPU read the caller's X-IP-Token (per-caller quota) for plain API calls, not only Gradio events."""
    token = LocalContext.request.set(gr.Request(request=request))
    try:
        return await call_next(request)
    finally:
        LocalContext.request.reset(token)


# ---------------------------------------------------------------- chat UI at "/" (the API lives at /api/v1)
_chat_sessions: dict = {}        # Gradio browser session -> TerraPulse session_id (keeps follow-ups in context)


async def chat_fn(message: str, history: list, request: gr.Request):
    pipeline = getattr(fastapi_app.state, "pipeline", None)
    if pipeline is None:
        yield "Service is still starting. Please try again in a few seconds."
        return
    key = getattr(request, "session_hash", None) or "anon"
    progress = []
    try:
        async for ev in pipeline.run(message.strip(), _chat_sessions.get(key)):
            if ev["event"] == "stage":
                progress.append("⏳ " + ev["data"]["message"])
                yield "\n".join(progress[-3:])
            elif ev["event"] == "final":
                data = ev["data"]
                _chat_sessions[key] = data.get("session_id")
                if len(_chat_sessions) > 5000:          # bounded: drop the oldest browser sessions
                    for old in list(_chat_sessions)[:1000]:
                        _chat_sessions.pop(old, None)
                md = data.get("answer_markdown", "")
                clar = data.get("clarification")
                if clar and clar.get("options"):
                    md += "\n\n" + " · ".join(f"*{o}*" for o in clar["options"][:6])
                g = (data.get("grounding") or {}).get("status")
                if g in ("passed", "repaired"):
                    md += f"\n\n<sub>✔ every number checked against live data · {round(data['timings_ms']['total'] / 1000)} s</sub>"
                yield md
    except Exception as exc:  # noqa: BLE001
        yield f"Sorry, something went wrong: {exc}"


demo = gr.ChatInterface(
    fn=chat_fn,
    type="messages",
    title="🌍 TerraPulse AI",
    description=("Where and when to head outdoors, grounded in live weather, air quality and OpenStreetMap. "
                 "Ask, then follow up: *why?* · *what about Sunday?* · *somewhere quieter?* "
                 "API: `POST /api/v1/chat` or `/api/v1/ask` · docs at `/docs`."),
    examples=["Hi! What can you do?", "Evening jog near Indiranagar Bengaluru tomorrow?",
              "I have mild asthma, any good spot for a morning walk near Malleshwaram tomorrow?",
              "Box cricket Saturday or Sunday near Thane?", "Kal subah jogging karna hai Salt Lake Kolkata mein"],
    cache_examples=False,
)

# Client-side rendering: Spaces set GRADIO_SSR_MODE=true, but SSR needs a separate Node server that a mounted app
# does not have (assets under /_app would 404).
app = gr.mount_gradio_app(fastapi_app, demo, path="/", ssr_mode=False)

if __name__ == "__main__":
    # ZeroGPU registers @spaces.GPU functions from gr.Blocks.launch(); we serve with uvicorn to keep the FastAPI
    # routes, so send the same startup report ourselves (after the model is on "cuda").
    try:
        from spaces import zero as _zero
        if hasattr(_zero, "startup"):
            _zero.startup()
    except Exception as exc:  # noqa: BLE001
        print("ZeroGPU startup report failed:", exc)
    uvicorn.run(app, host="0.0.0.0", port=7860)  # the object, not "app:app" (that string resolves to app/)
