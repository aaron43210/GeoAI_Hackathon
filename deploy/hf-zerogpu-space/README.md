---
title: TerraPulse AI
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---

# 🌍 TerraPulse AI — GeoAI Outdoor Activity Advisor

Official Live API & Web Service for **GeoAI Hackathon @ CSRE, IIT Bombay**.

TerraPulse AI answers free-form questions like *"I have mild asthma, where's good for a morning walk near Powai in the next two days?"* with grounded **place × time** recommendations and practical advice.

> **Tip for evaluators:** the endpoint runs on a Hugging Face ZeroGPU Space, where free GPU runs are metered per
> caller. Adding your own Hugging Face token (`-H "Authorization: Bearer hf_..."`, read access is enough) moves you to
> your account's larger quota, so every answer is written by the fine-tuned model. Without it, once an IP's free runs
> are used up the API still answers from the same live data via its rule parser and data template.

### Live Endpoints:
- `POST /api/v1/ask` (JSON) · `GET /api/v1/ask?q=...&format=text` (plain text) · `GET /api/v1/ask/stream?q=...` (SSE)
- `POST /api/v1/recommend`: JSON response with scored venue-hours, weather, and evidence
- `GET /api/v1/health`: Live health status and SLM model info
- `GET /docs`: Interactive Swagger OpenAPI documentation
