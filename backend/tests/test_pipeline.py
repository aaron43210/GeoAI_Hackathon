"""End-to-end pipeline with all upstream HTTP mocked (respx). Fixture data is test-only and clearly fictional."""
import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

OVERPASS = "https://overpass.test/api/interpreter"
LAT, LON = 10.0, 76.0


def _hours():
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    start = datetime.combine(today - timedelta(days=1), datetime.min.time())
    return [start + timedelta(hours=h) for h in range(24 * 5)]


def _forecast(n_points: int, hot_hour: int = 13):
    hours = _hours()
    def point(i):
        hourly = {"time": [h.strftime("%Y-%m-%dT%H:%M") for h in hours]}
        hourly["temperature_2m"] = [24 + (6 if 11 <= h.hour <= 15 else 0) for h in hours]
        hourly["apparent_temperature"] = [26 + (14 if 11 <= h.hour <= 15 else 0) for h in hours]
        hourly["relative_humidity_2m"] = [60 for _ in hours]
        hourly["precipitation_probability"] = [5 for _ in hours]
        hourly["precipitation"] = [0.0 for _ in hours]
        hourly["weather_code"] = [1 for _ in hours]
        hourly["cloud_cover"] = [30 for _ in hours]
        hourly["visibility"] = [20000 for _ in hours]
        hourly["wind_speed_10m"] = [10 for _ in hours]
        hourly["wind_direction_10m"] = [90 if h.hour < 12 else 270 for h in hours]
        hourly["wind_gusts_10m"] = [18 for _ in hours]
        hourly["uv_index"] = [0 if h.hour < 7 or h.hour > 17 else 6 for h in hours]
        hourly["is_day"] = [1 if 6 <= h.hour <= 18 else 0 for h in hours]
        days = sorted({h.date() for h in hours})
        daily = {"time": [d.isoformat() for d in days], "sunrise": [f"{d}T06:30" for d in days],
                 "sunset": [f"{d}T18:10" for d in days], "uv_index_max": [8] * len(days),
                 "precipitation_sum": [0] * len(days)}
        return {"latitude": LAT + i * 0.01, "longitude": LON, "timezone": "Asia/Kolkata", "utc_offset_seconds": 19800,
                "hourly": hourly, "daily": daily}
    data = [point(i) for i in range(n_points)]
    return data if n_points > 1 else data[0]


def _air(n_points: int):
    hours = _hours()
    def point(i):
        return {"latitude": LAT, "longitude": LON, "hourly": {
            "time": [h.strftime("%Y-%m-%dT%H:%M") for h in hours],
            "pm2_5": [20 for _ in hours], "pm10": [40 for _ in hours], "nitrogen_dioxide": [10 for _ in hours],
            "ozone": [40 for _ in hours], "us_aqi": [60 for _ in hours], "us_aqi_pm2_5": [60 for _ in hours],
            "us_aqi_pm10": [30 for _ in hours], "us_aqi_ozone": [30 for _ in hours],
            "us_aqi_nitrogen_dioxide": [5 for _ in hours]}}
    data = [point(i) for i in range(n_points)]
    return data if n_points > 1 else data[0]


CANDIDATES = {"elements": [
    {"type": "way", "id": 101, "tags": {"leisure": "park", "name": "Testville Green"},
     "bounds": {"minlat": 10.004, "minlon": 76.004, "maxlat": 10.006, "maxlon": 76.007}},
    {"type": "way", "id": 102, "tags": {"leisure": "park", "name": "Riverside Test Park"},
     "bounds": {"minlat": 10.02, "minlon": 75.99, "maxlat": 10.023, "maxlon": 75.994}},
]}
CONTEXT = {"elements": [
    {"type": "way", "id": 900, "tags": {"highway": "trunk", "name": "Test Highway"},
     "geometry": [{"lat": 10.005, "lon": 76.009}, {"lat": 10.010, "lon": 76.009}]},
    {"type": "way", "id": 901, "tags": {"leisure": "park", "name": "Testville Green"},
     "geometry": [{"lat": 10.004, "lon": 76.004}, {"lat": 10.004, "lon": 76.007}, {"lat": 10.006, "lon": 76.007},
                  {"lat": 10.006, "lon": 76.004}, {"lat": 10.004, "lon": 76.004}]},
]}
NOMINATIM = [{"lat": str(LAT), "lon": str(LON), "name": "Testville", "display_name": "Testville, Test District, Kerala, India",
              "boundingbox": ["9.9", "10.1", "75.9", "76.1"], "importance": 0.5, "type": "suburb",
              "address": {"city": "Testville", "state": "Kerala"}}]


def _mock(router, overpass_ok=True, weather_ok=True, aq_ok=True, candidates=None):
    router.get(url__regex=r"https://nominatim\.openstreetmap\.org/search.*").mock(
        side_effect=lambda req: httpx.Response(200, json=NOMINATIM if "viewbox" not in str(req.url) else [
            {"lat": "10.005", "lon": "76.005", "name": "Nominatim Test Garden", "osm_type": "way", "osm_id": 555,
             "category": "leisure", "type": "park", "boundingbox": ["10.004", "10.006", "76.004", "76.007"]}]))

    def overpass(req):
        if not overpass_ok:
            return httpx.Response(504)
        body = req.content.decode()
        return httpx.Response(200, json=CONTEXT if "around.c" in body else (candidates or CANDIDATES))
    router.post(OVERPASS).mock(side_effect=overpass)

    def wx(req):
        if not weather_ok:
            return httpx.Response(503)
        n = len(req.url.params["latitude"].split(","))
        return httpx.Response(200, json=_forecast(n))
    router.get(url__regex=r"https://api\.open-meteo\.com/v1/forecast.*").mock(side_effect=wx)
    router.get(url__regex=r"https://air-quality-api\.open-meteo\.com/v1/air-quality.*").mock(
        side_effect=lambda req: httpx.Response(200, json=_air(len(req.url.params["latitude"].split(","))))
        if aq_ok else httpx.Response(503))

    def osrm(req):
        n = str(req.url).split("/driving/")[1].split("?")[0].count(";")
        return httpx.Response(200, json={"code": "Ok", "durations": [[0] + [300.0 * (i + 1) for i in range(n)]],
                                         "distances": [[0] + [2000.0 * (i + 1) for i in range(n)]]})
    router.get(url__regex=r"https://router\.project-osrm\.org/table/.*").mock(side_effect=osrm)


def _ask(q):
    from app.main import app
    with TestClient(app) as c:
        return c.post("/api/v1/ask", json={"question": q}).json()


@respx.mock(assert_all_called=False)
def test_happy_path_is_grounded(respx_mock):
    _mock(respx_mock)
    r = _ask("Where can I go running tomorrow evening near Testville?")
    assert r["status"] in ("ok", "partial"), r.get("answer_markdown")
    rec = r["recommendation"]
    assert rec["venue"]["name"] in ("Testville Green", "Riverside Test Park")
    assert 16 <= datetime.fromisoformat(rec["start"]).hour <= 19
    assert r["grounding"]["status"] == "templated"            # SLM disabled in tests → template path
    ids = {e["id"] for e in r["evidence"]}
    assert set(re.findall(r"\[(E\d+)\]", r["answer_markdown"])) <= ids
    assert any("Test Highway" in (e.get("label") or "") or "Test Highway" in r["fact_sheet"] for e in r["evidence"])
    assert r["heatmap"]["venues"] and r["heatmap"]["scores"]


@respx.mock(assert_all_called=False)
def test_midday_heat_is_avoided(respx_mock):
    _mock(respx_mock)
    r = _ask("running tomorrow near Testville")
    start = datetime.fromisoformat(r["recommendation"]["start"])
    assert not (11 <= start.hour <= 15)                        # feels-like 38 °C gated for running
    assert r["avoid"]


@respx.mock(assert_all_called=False)
def test_overpass_down_uses_nominatim_not_fiction(respx_mock):
    _mock(respx_mock, overpass_ok=False)
    r = _ask("picnic tomorrow afternoon near Testville")
    names = [v["name"] for v in r["venues"]]
    assert names == ["Nominatim Test Garden"]
    assert any("Nominatim" in lim for lim in r["limitations"])


@respx.mock(assert_all_called=False)
def test_weather_down_refuses_to_recommend(respx_mock):
    _mock(respx_mock, weather_ok=False)
    r = _ask("running tomorrow evening near Testville")
    assert r["status"] == "error" and "recommendation" not in r


def test_missing_location_asks():
    r = _ask("Is it okay to cycle to work at 8 AM?")
    assert r["status"] == "clarification_needed" and r["clarification"]["field"] == "location"


def test_out_of_scope():
    assert _ask("can you recommend a good stock to buy")["status"] == "out_of_scope"


class _QuotaExhaustedSLM:
    """Behaves like ZeroGPU after the caller's free runs are used up."""
    ready = True

    def complete(self, *a, **k):
        raise RuntimeError("You have exceeded your ZeroGPU runs limit")

    def stream(self, *a, **k):
        raise RuntimeError("You have exceeded your ZeroGPU runs limit")
        yield  # pragma: no cover

    def describe(self):
        return {"backend": "stub", "loaded": True, "path_used": "finetuned"}


@respx.mock(assert_all_called=False)
def test_gpu_quota_exhausted_still_answers(respx_mock):
    _mock(respx_mock)
    from app.main import app
    with TestClient(app) as c:
        app.state.pipeline.slm = _QuotaExhaustedSLM()
        r = c.post("/api/v1/ask", json={"question": "running tomorrow evening near Testville"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("ok", "partial") and body["grounding"]["status"] == "templated"
        assert body["intent"]["path"] == "rules (SLM failed)" and body["answer_markdown"]
