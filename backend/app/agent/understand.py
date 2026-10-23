"""Question → Intent. SLM first (grammar-constrained JSON), cross-checked with the rule parser.

Merge policy is safety-monotone: health/age profiles are the union of both (profiles only tighten limits), and
rule-detected values fill gaps the SLM left. Nothing is invented: unresolved fields stay empty and trigger a
clarifying question downstream.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Tuple

from app.agent import rule_parser
from app.schemas.intent import Intent, intent_json_schema
from app.slm.prompts import intent_messages
from app.slm.runtime import SLMRuntime

logger = logging.getLogger(__name__)


async def understand(question: str, slm: SLMRuntime, previous: Optional[Dict[str, Any]] = None
                     ) -> Tuple[Intent, str]:
    rules = rule_parser.parse(question)
    if not slm.ready:
        return rules, "rules"
    try:
        raw = await asyncio.to_thread(slm.complete, intent_messages(question, previous),
                                      json_schema=intent_json_schema(), max_tokens=slm.settings.SLM_MAX_TOKENS_INTENT,
                                      temperature=0.0)
        model = Intent.from_model_output(json.loads(raw)).normalised()
    except Exception as exc:  # noqa: BLE001 - fall back to rules, and say so in the response
        logger.warning("SLM intent extraction failed (%s); using rule parser", exc)
        return rules, "rules (SLM failed)"
    return merge_intents(question, rules, model), "slm+rules"


def merge_intents(question: str, rules: Intent, model: Intent) -> Intent:
    """Combine the model's intent with the rule parser's (safety-monotone, see module docstring)."""
    data = model.model_dump()
    data["profiles"] = sorted(set(model.profiles) | set(rules.profiles))
    for key in ("activity", "location_text", "radius_km", "max_travel_min", "duration_min", "age"):
        if data[key] in (None, "") and getattr(rules, key) not in (None, ""):
            data[key] = getattr(rules, key)
    if not data["time_text"]:
        data["time_text"] = question
    data["preferences"] = sorted(set(model.preferences) | set(rules.preferences))
    data["use_user_location"] = model.use_user_location or rules.use_user_location
    if not rules.in_scope and rules.activity is None:
        data["in_scope"], data["out_of_scope_reason"] = False, rules.out_of_scope_reason
    elif rules.activity is not None or rule_parser.has_in_scope_cue(question):
        data["in_scope"], data["out_of_scope_reason"] = True, None
    return Intent(**data).normalised()
