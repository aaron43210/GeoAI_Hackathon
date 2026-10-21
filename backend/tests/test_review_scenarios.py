"""Regression suite from the external review: safety decisions are made by code and survive the language model."""
import copy
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import respx
from fastapi.testclient import TestClient

from tests.test_pipeline import CANDIDATES, _mock


class _LyingSLM:
    """Writes a cheerful headline and reassurance regardless of the data."""
    ready = True

    def __init__(self, text):
        self.text = text

    def stream(self, *a, **k):
        yield self.text

    def complete(self, *a, **k):
        return self.text

    def describe(self):
        return {"backend": "stub", "loaded": True, "path_used": "finetuned"}


def _client():
    from app.main import app
    return app, TestClient(app)


@respx.mock(assert_all_called=False)
def test_missing_air_quality_is_disclosed_not_assumed_clean(respx_mock):
    _mock(respx_mock, aq_ok=False)
    app, c = _client()
    with c:
        r = c.post("/api/v1/ask", json={"question": "running tomorrow evening near Testville"}).json()
    assert r["status"] == "partial"
    assert any("air-quality" in l for l in r["limitations"])
    assert "air quality forecast: unavailable" in r["fact_sheet"]
    assert "US AQI up to" not in r["answer_markdown"]


@respx.mock(assert_all_called=False)
def test_access_no_venue_is_excluded(respx_mock):
    cands = copy.deepcopy(CANDIDATES)
    cands["elements"][0]["tags"]["access"] = "no"
    _mock(respx_mock, candidates=cands)
    app, c = _client()
    with c:
        r = c.post("/api/v1/ask", json={"question": "running tomorrow evening near Testville"}).json()
    assert "Testville Green" not in [v["name"] for v in r["venues"]]


@respx.mock(assert_all_called=False)
def test_opening_hours_not_claimed_and_alternatives_are_eligible(respx_mock):
    _mock(respx_mock)
    app, c = _client()
    with c:
        r = c.post("/api/v1/ask", json={"question": "running tomorrow near Testville"}).json()
    assert any("opening hours" in l for l in r["limitations"])
    assert all(a["status"] != "avoid" for a in r["alternatives"])
    assert r["recommendation"]["status"] != "avoid"


@respx.mock(assert_all_called=False)
def test_model_cannot_overwrite_the_decision(respx_mock):
    _mock(respx_mock)
    app, c = _client()
    with c:
        app.state.pipeline.slm = _LyingSLM("**Best pick:** Somewhere Lovely, perfect all day!\n**Why:** great weather.")
        r = c.post("/api/v1/ask", json={"question": "running tomorrow evening near Testville"}).json()
    assert "Somewhere Lovely" not in r["answer_markdown"]
    assert r["answer_markdown"].startswith("**Best pick:** [E")       # code-rendered decision line


@respx.mock(assert_all_called=False)
def test_conversation_greeting_followup_and_why(respx_mock):
    _mock(respx_mock)
    app, c = _client()
    with c:
        hi = c.post("/api/v1/chat", json={"message": "hi"}).json()
        assert hi["status"] == "chat" and "TerraPulse" in hi["reply"]
        sid = hi["session_id"]
        plan = c.post("/api/v1/chat", json={"message": "running tomorrow evening near Testville", "session_id": sid}).json()
        assert plan["status"] in ("ok", "partial") and plan["recommendation"]
        why = c.post("/api/v1/chat", json={"message": "why that time?", "session_id": sid}).json()
        assert why["status"] == "explanation" and "[E" in why["reply"]
        later = c.post("/api/v1/chat", json={"message": "what about the day after tomorrow?", "session_id": sid}).json()
        assert later["status"] in ("ok", "partial", "not_recommended")
        assert later["understood_as"]["activity"] == "running"
        day2 = (datetime.now(ZoneInfo("Asia/Kolkata")).date() + timedelta(days=2)).isoformat()
        assert later["recommendation"]["label"] and day2[-2:] in later["recommendation"]["label"]


def test_fallback_explanation_includes_the_reason():
    from app.agent.conversation import _fallback_explanation
    sheet = ("RESULT: NOT RECOMMENDED in the requested time: every hour hits a safety limit. Least-bad option: [E1] X\n"
             "FACTS:\n[E4] rain probability up to 100 % (17:00–18:00)\nAVOID / TIMING:\n"
             "[E14] Thu 17:00–20:00: rain probability 100% is above 80%\nLIMITATIONS: none")
    out = _fallback_explanation(sheet, "why is it not recommended?")
    assert "NOT RECOMMENDED" in out and "rain probability 100% is above 80%" in out
