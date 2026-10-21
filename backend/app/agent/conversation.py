"""Conversation layer: small talk, help, and follow-up questions about the previous answer.

Plan requests ("run near X tomorrow", "what about Sunday?") go to the planning pipeline. Questions about the last
answer ("why not 8 am?", "where does the AQI come from?") are answered by the SLM from that answer's fact sheet and
pass the same grounding verifier; if they fail, the relevant fact-sheet lines are shown instead.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.agent.rule_parser import detect_activity
from app.evidence.grounding import verify
from app.knowledge import get_knowledge
from app.slm.prompts import REPAIR_NOTE, explain_messages
from app.slm.runtime import SLMRuntime, strip_think

logger = logging.getLogger(__name__)


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def _has_phrase(text: str, phrases: List[str]) -> bool:
    t = " " + " ".join(_words(text)) + " "
    return any(f" {p} " in t for p in phrases)


def classify_turn(text: str, has_last_answer: bool) -> str:
    """greeting | thanks | help | explain | plan"""
    conv = get_knowledge().language["conversation"]
    words = _words(text)
    short = len(words) <= conv["smalltalk_max_words"]
    activity, _ = detect_activity(text.lower())
    if _has_phrase(text, conv["help"]) and activity is None:
        return "help"
    if short and activity is None and _has_phrase(text, conv["thanks"]):
        return "thanks"
    if short and activity is None and _has_phrase(text, conv["greetings"]):
        return "greeting"
    if has_last_answer and activity is None and words and any(
            " ".join(words).startswith(p) for p in conv["explain_starts"]):
        return "explain"
    return "plan"


def smalltalk_reply(kind: str) -> str:
    kb = get_knowledge()
    acts = ", ".join(a.label.split(" (")[0].split(" / ")[0].lower() for a in kb.activities.values())
    if kind == "thanks":
        return "You're welcome! Ask me about another plan any time, or follow up with *“what about Sunday?”* or *“somewhere closer?”*."
    intro = ("Hi! I'm **TerraPulse**, an outdoor-activity assistant. " if kind == "greeting" else "")
    return (intro +
            "Tell me what you want to do, where and roughly when, and I'll check live weather, air quality, "
            "OpenStreetMap places and travel times across several nearby places and hours, then suggest the best "
            "place × time with what to carry, what to watch for and when to avoid going.\n\n"
            f"I can plan: {acts}.\n\n"
            "Mention people's needs too (asthma, kids, seniors, pregnancy, heart condition). Try:\n"
            "- *Evening run near ‹your area› tomorrow?*\n"
            "- *Kids' football this weekend within 10 km of ‹your area›*\n"
            "- *Gentle morning walk for my parents in ‹your city› on Sunday*\n\n"
            "Afterwards you can ask *“why?”*, *“what about Saturday?”* or *“somewhere quieter?”*.")


def _fallback_explanation(sheet: str, question: str) -> str:
    """Deterministic answer: the decision, its avoid/timing reasons, then the facts closest to the question."""
    lines = sheet.splitlines()
    decision = [ln for ln in lines if ln.startswith(("BEST:", "RESULT:", "HEALTH WARNING:"))]
    reasons, in_avoid = [], False
    for ln in lines:
        if ln.startswith("AVOID / TIMING:"):
            in_avoid = True
            continue
        if in_avoid:
            if ln.startswith("[E"):
                reasons.append(ln)
            else:
                break
    q = set(_words(question)) - set(get_knowledge().language["stop_words"]) - {"why", "what", "how", "not"}
    facts = [ln for ln in lines if ln.startswith("[E") and ln not in reasons]
    related = [ln for ln in sorted(facts, key=lambda ln: -len(q & set(_words(ln)))) if q & set(_words(ln))][:3]
    shown = decision + reasons + related
    return "Here is what the data behind my last answer shows:\n" + "\n".join(f"- {ln}" for ln in shown)


async def explain(question: str, last: Dict[str, Any], slm: SLMRuntime) -> Tuple[str, Dict[str, Any], str]:
    sheet = last["fact_sheet"]
    evidence = last.get("evidence") or []
    ids = {e["id"] for e in evidence}
    values = {e["id"]: e.get("value") for e in evidence}
    names = sorted({e.get("venue") for e in evidence if e.get("venue")} |
                   {e.get("label") for e in evidence if e.get("kind") == "osm"})
    if slm.ready:
        try:
            draft = strip_think(await asyncio.to_thread(slm.complete, explain_messages(sheet, question),
                                                        max_tokens=260))
            g = verify(draft, sheet, ids, names, None, True, values)
            if draft.strip() and g.status == "passed":
                return draft, {"status": "passed", "facts_checked": g.facts_checked, "violations": []}, "slm"
            logger.info("explanation failed grounding: %s", g.violations)
            if draft.strip():
                msgs = explain_messages(sheet, question) + [
                    {"role": "assistant", "content": draft},
                    {"role": "user", "content": REPAIR_NOTE.format(violations="; ".join(g.violations))}]
                fixed = strip_think(await asyncio.to_thread(slm.complete, msgs, max_tokens=260))
                g2 = verify(fixed, sheet, ids, names, None, True, values)
                if fixed.strip() and g2.status == "passed":
                    return fixed, {"status": "repaired", "first_violations": g.violations,
                                   "facts_checked": g2.facts_checked, "violations": []}, "slm (repaired)"
        except Exception as exc:  # noqa: BLE001
            logger.warning("SLM explanation failed (%s); using fact-sheet lines", exc)
    return _fallback_explanation(sheet, question), {"status": "templated"}, "template"
