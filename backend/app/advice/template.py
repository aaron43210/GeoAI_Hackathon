"""Deterministic answer renderer, used when the SLM is unavailable or its draft fails grounding.

It renders straight from the fact sheet sections, so it can only ever say what the fact sheet says.
"""
from __future__ import annotations

from typing import Dict, List


def _section(lines: List[str], header: str) -> List[str]:
    out, on = [], False
    for ln in lines:
        if ln.startswith(header):
            on = True
            rest = ln[len(header):].strip()
            if rest:
                out.append(rest)
            continue
        if on:
            if ln.isupper() or (ln.split(":")[0].isupper() and ":" in ln):
                break
            out.append(ln)
    return out


def _heads(lines: List[str]) -> Dict[str, str]:
    head: Dict[str, str] = {}
    for ln in lines:
        key = ln.split(":", 1)[0]
        if key in ("BEST", "RESULT", "CARRY", "WATCH FOR", "LIMITATIONS", "HEALTH WARNING"):
            head[key] = ln.split(":", 1)[1].strip()
    return head


def headline(sheet_text: str) -> str:
    """The decision itself, rendered by code: the language model never writes (or softens) it."""
    head = _heads(sheet_text.splitlines())
    parts: List[str] = []
    if "BEST" in head:
        parts.append(f"**Best pick:** {head['BEST']}")
    elif "RESULT" in head:
        result = head["RESULT"].replace("NOT RECOMMENDED in the requested time: ", "")
        parts.append(f"**Not recommended:** {result}")
    if "HEALTH WARNING" in head:
        parts.append(f"⚠️ **Health warning:** {head['HEALTH WARNING']}")
    return "  \n".join(parts)


def body_of(draft: str) -> str:
    """Drop any headline the model wrote itself; keep its explanation (from **Why** onwards)."""
    idx = draft.lower().find("**why")
    if idx >= 0:
        return draft[idx:].strip()
    keep = [ln for ln in draft.splitlines()
            if not ln.lstrip().startswith(("**Best pick", "**Not recommended", "Best pick", "Not recommended",
                                           "⚠", "**Heads-up", "**Health warning"))]
    return "\n".join(keep).strip()


def render(sheet_text: str) -> str:
    lines = sheet_text.splitlines()
    head = _heads(lines)
    parts: List[str] = [headline(sheet_text)] if headline(sheet_text) else []
    facts = _section(lines, "FACTS:")
    if facts:
        parts.append("**Why (live data):**\n" + "\n".join(f"- {f}" for f in facts))
    if "CARRY" in head:
        parts.append("**Carry:**\n" + "\n".join(f"- {x.strip()}" for x in head["CARRY"].split(";")))
    if "WATCH FOR" in head:
        parts.append("**Watch for:**\n" + "\n".join(f"- {x.strip()}" for x in head["WATCH FOR"].split(";")))
    avoid = _section(lines, "AVOID / TIMING:")
    if avoid:
        parts.append("**Avoid / timing:**\n" + "\n".join(f"- {a}" for a in avoid))
    others = _section(lines, "OTHER OPTIONS:")
    if others:
        parts.append("**Other options:**\n" + "\n".join(f"- {o}" for o in others))
    notes = [ln[len("NOTE: "):] for ln in lines if ln.startswith("NOTE: ")]
    if notes:
        parts.append("_" + " ".join(notes) + "_")
    if "LIMITATIONS" in head:
        parts.append(f"_Limitations: {head['LIMITATIONS']}_")
    return "\n\n".join(parts)
