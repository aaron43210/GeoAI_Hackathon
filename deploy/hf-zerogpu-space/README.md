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

### Live Endpoints:
- `POST /api/v1/ask` (JSON) · `GET /api/v1/ask?q=...&format=text` (plain text) · `GET /api/v1/ask/stream?q=...` (SSE)
- `POST /api/v1/recommend`: JSON response with scored venue-hours, weather, and evidence
- `GET /api/v1/health`: Live health status and SLM model info
- `GET /docs`: Interactive Swagger OpenAPI documentation
