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


# ---------------------------------------------------------------- small web demo at "/" (the API lives at /api/v1)
async def ask_terrapulse(question: str):
    if not question or not question.strip():
        return "Please enter a question, e.g. *'evening run near Baner tomorrow'*"
    pipeline = getattr(fastapi_app.state, "pipeline", None)
    if pipeline is None:
        return "Service is still starting. Please try again in a few seconds."
    try:
        async for ev in pipeline.run(question.strip()):
            if ev.get("event") == "final":
                data = ev.get("data", {})
                md = data.get("answer_markdown", "")
                if data.get("sources"):
                    md += "\n\n---\n**Data sources:** " + " · ".join(data["sources"])
                return md
        return "No answer was produced. Please rephrase the question."
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


with gr.Blocks(title="TerraPulse AI") as demo:
    gr.Markdown("# 🌍 TerraPulse AI\nWhere and when to head outdoors, grounded in live open data. "
                "API: `POST /api/v1/ask` · docs at `/docs`.")
    with gr.Row():
        inp = gr.Textbox(label="Your question", placeholder="e.g. Kids' football practice near Koramangala, "
                         "Saturday or Sunday?", lines=2, scale=4)
        btn = gr.Button("Ask", variant="primary", scale=1)
    out = gr.Markdown()
    gr.Examples(examples=[["Evening jog near Indiranagar Bengaluru tomorrow?"],
                          ["I have mild asthma, any good spot for a morning walk near Malleshwaram tomorrow?"],
                          ["Box cricket Saturday or Sunday near Thane?"],
                          ["Kal subah jogging karna hai Salt Lake Kolkata mein"]], inputs=inp)
    btn.click(ask_terrapulse, inputs=inp, outputs=out)
    inp.submit(ask_terrapulse, inputs=inp, outputs=out)

app = gr.mount_gradio_app(fastapi_app, demo, path="/")

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
