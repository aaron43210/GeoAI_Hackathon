import os
import sys
from pathlib import Path

os.environ.setdefault("SLM_ENABLED", "false")
os.environ.setdefault("WAQI_TOKEN", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import pytest  # noqa: E402

OVERPASS_TEST_MIRROR = "https://overpass.test/api/interpreter"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Offline, deterministic settings for every test: fictional Overpass mirror, no model."""
    monkeypatch.setenv("OVERPASS_MIRRORS", OVERPASS_TEST_MIRROR)
    monkeypatch.setenv("SLM_ENABLED", "false")
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
