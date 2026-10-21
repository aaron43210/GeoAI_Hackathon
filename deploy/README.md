# Deployment

| Target | Files | Model runtime |
|---|---|---|
| **Hugging Face ZeroGPU Space** (live endpoint) | [`hf-zerogpu-space/`](hf-zerogpu-space/) | merged fine-tune with transformers on ZeroGPU (`@spaces.GPU`) |
| Docker (any VM, e.g. Oracle Always Free, or an HF CPU Space) | [`../Dockerfile`](../Dockerfile), [`HF_SPACE_README.md`](HF_SPACE_README.md) | GGUF Q4_K_M with llama.cpp on CPU |

To refresh the Space: copy `backend/app/` to the Space repo as `app/`, add the three files from
`hf-zerogpu-space/`, and push. ZeroGPU containers have no usable CPU budget for inference, which is why the Space runs
the model on the GPU; if GPU quota is exhausted the API still answers via its rule parser and data template.
