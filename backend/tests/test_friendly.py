"""Friendly chat answers: plain text first, sources last, same decision and facts as the verified version."""
import re

import respx
from fastapi.testclient import TestClient

from tests.test_pipeline import _mock

_ECITE = re.compile(r"\[E\d+\]")
_FOOT = re.compile(r"\[(\d+)\]")


def _ask(q):
    from app.main import app
    with TestClient(app) as c:
        return c.post("/api/v1/ask", json={"question": q}).json()


@respx.mock(assert_all_called=False)
def test_chat_answer_is_friendly_with_sources_at_the_end(respx_mock):
    _mock(respx_mock)
    r = _ask("running tomorrow evening near Testville")
    chat = r["answer_markdown"]
    body, _, data = chat.partition("**The data behind this**")
    assert data, "data section missing"
    assert not _ECITE.search(chat), "internal evidence ids leaked into the chat answer"
    refs = set(_FOOT.findall(body))
    listed = set(re.findall(r"^\[(\d+)\]", data, flags=re.M))
    assert refs and refs <= listed                       # every footnote in the text has a source line
    assert r["recommendation"]["venue"]["name"] in body.split("\n")[0]
    assert "tomorrow" in body.split("\n")[0]             # human dates, not "Thu 22 Oct"


@respx.mock(assert_all_called=False)
def test_friendly_numbers_come_from_the_evidence(respx_mock):
    _mock(respx_mock)
    r = _ask("running tomorrow evening near Testville")
    body = r["answer_markdown"].partition("---")[0]
    body = _FOOT.sub(" ", body)
    from app.evidence.grounding import _NUM, _TIME, _AMPM, _variants
    sheet_nums = set()
    for m in _NUM.finditer(r["fact_sheet"]):
        sheet_nums |= _variants(m.group(1))
    stripped = _AMPM.sub(" ", _TIME.sub(" ", body))
    stripped = re.sub(r"\d{1,2}(?::\d{2})?–\d{1,2}(?::\d{2})?\s*(?:am|pm)", " ", stripped)  # human time ranges
    unknown = [m.group(1) for m in _NUM.finditer(stripped) if m.group(1) not in sheet_nums]
    assert not unknown, f"numbers not in the fact sheet: {unknown}"


def test_human_window_and_bands():
    from datetime import datetime
    from app.advice.friendly import band, human_window
    from app.knowledge import get_knowledge
    now = datetime(2026, 10, 21, 9, 0)
    assert human_window(datetime(2026, 10, 22, 6), datetime(2026, 10, 22, 8), now) == "tomorrow 6–8 am"
    assert human_window(datetime(2026, 10, 24, 11), datetime(2026, 10, 24, 14), now) == "on Saturday 11 am–2 pm"
    ph = get_knowledge().phrasing
    assert band(166, ph["aqi_bands"]) == "unhealthy" and band(45, ph["aqi_bands"]) == "good"
