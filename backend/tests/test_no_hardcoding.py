"""Guards for the hackathon rules: no fabricated-data fallbacks and no example-specific logic in runtime code."""
import re
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
RUNTIME_FILES = [p for p in APP.rglob("*") if p.suffix in (".py", ".yaml")]

# Places from the hackathon's example questions must not appear in runtime code or knowledge files.
EXAMPLE_PLACES = ["koregaon", "whitefield", "powai", "hinjewadi", "andheri", "bandra", "lonavala", "bengaluru",
                  "bangalore", "pune", "mumbai"]
FABRICATION_WORDS = ["synthetic", "simulated", "baseline fallback", "mock", "fake", "dummy"]


def test_no_example_places_in_runtime():
    for p in RUNTIME_FILES:
        text = p.read_text(encoding="utf-8").lower()
        for place in EXAMPLE_PLACES:
            assert not re.search(r"\b" + place + r"\b", text), f"{place} hard-coded in {p.relative_to(APP)}"


def test_no_fabrication_code_paths():
    for p in RUNTIME_FILES:
        text = p.read_text(encoding="utf-8").lower()
        for w in FABRICATION_WORDS:
            assert w not in text, f"'{w}' found in {p.relative_to(APP)}"


def test_no_hosted_llm_clients():
    for p in RUNTIME_FILES:
        text = p.read_text(encoding="utf-8").lower()
        for lib in ("openai", "anthropic", "google.generativeai", "gemini", "openrouter"):
            assert lib not in text, f"{lib} referenced in {p.relative_to(APP)}"


def test_model_is_small():
    from app.config import Settings
    s = Settings()
    assert "phi-3" not in s.SLM_BASE_REPO.lower() and "1.7b" in s.SLM_BASE_FILE.lower()
