"""Grounding verifier: rejects an SLM answer that states anything the fact sheet does not support.

Checks (deterministic, vocabulary in knowledge/grounding.yaml):
  1 citations  every [E#] exists in the ledger, and at least one is present
  2 times      every clock time (HH:MM, "5 pm") appears in the fact sheet
  3 quantities every "number + unit" appears in the sheet with the same unit family; bare numbers must appear too
  4 names      multi-word capitalised names must match a known venue/area or appear in the sheet
  5 decision   the recommended (or least-bad) place is mentioned; a not-recommended result is framed as such
  6 severity   a value the sheet flags (e.g. AQI above the limit) is never described positively
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz

from app.knowledge import get_knowledge

_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.]\d)")
_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_AMPM = re.compile(r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(am|pm|a\.m\.|p\.m\.)", re.I)
_CITE = re.compile(r"\[(E\d+)\]")
_NAME = re.compile(r"(?<![.!?:*]\s)(?<!^)\b([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|of|the|de)){1,5})")


@dataclass
class GroundingResult:
    status: str                      # passed | failed
    violations: List[str] = field(default_factory=list)
    facts_checked: int = 0


def _variants(num: str) -> Set[str]:
    f = float(num)
    out = {num, str(int(round(f))), f"{f:.1f}".rstrip("0").rstrip(".")}
    if "." in num:
        out.add(str(int(f)))
    return out


def _unit_regex() -> Tuple[re.Pattern, Dict[str, str]]:
    fam: Dict[str, str] = {}
    for family, toks in get_knowledge().grounding["units"].items():
        for t in toks:
            fam[t] = family
    alts = "|".join(re.escape(t) for t in sorted(fam, key=len, reverse=True))
    return re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(" + alts + r")(?![a-z])", re.I), fam


def _quantities(text: str) -> Set[Tuple[str, str]]:
    rx, fam = _unit_regex()
    out = set()
    for m in rx.finditer(text):
        for v in _variants(m.group(1)):
            out.add((v, fam[m.group(2).lower()]))
    return out


def _numeric(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, (list, tuple)):
        return all(isinstance(x, (int, float)) for x in v)
    if isinstance(v, dict):
        return any(isinstance(x, (int, float)) for x in v.values())
    return False


def _comparison_errors(text: str, cfg: Optional[Dict[str, object]]) -> List[str]:
    """Check 'A <exceeds|above|below|within> B' clauses: the stated order of the two numbers must be true."""
    if not cfg:
        return []
    out: List[str] = []
    gap = int(cfg.get("max_gap_chars", 60))
    for kind, words in (("greater", cfg.get("greater", [])), ("lower", cfg.get("lower", []))):
        alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
        rx = re.compile(r"(\d+(?:\.\d+)?)[^.;\n\d]{0,%d}?\b(?:%s)\b[^.;\n\d]{0,%d}?(\d+(?:\.\d+)?)" % (gap, alt, gap), re.I)
        for m in rx.finditer(text):
            a, b = float(m.group(1)), float(m.group(2))
            if (kind == "greater" and not a > b) or (kind == "lower" and not a <= b):
                out.append(f"false comparison: '{m.group(0).strip()[:70]}'")
    return out


def _time_comparison_errors(text: str, cfg: Optional[Dict[str, object]]) -> List[str]:
    if not cfg:
        return []
    out: List[str] = []
    gap = int(cfg.get("max_gap_chars", 60))
    t = r"([01]?\d|2[0-3]):([0-5]\d)"
    for kind, words in (("before", cfg.get("time_before", [])), ("after", cfg.get("time_after", []))):
        alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
        rx = re.compile(t + r"(?:\s*[–-]\s*" + t + r")?[^.;\n]{0,%d}?\b(?:%s)\b[^.;\n\d]{0,%d}?" % (gap, alt, gap) + t, re.I)
        for m in rx.finditer(text):
            first = int(m.group(1)) * 60 + int(m.group(2))
            last = int(m.group(3)) * 60 + int(m.group(4)) if m.group(3) else first
            ref = int(m.group(5)) * 60 + int(m.group(6))
            if (kind == "before" and not last <= ref) or (kind == "after" and not first >= ref):
                out.append(f"false time comparison: '{m.group(0).strip()[:70]}'")
    return out


def _band_errors(text: str, checks: List[Dict[str, object]]) -> List[str]:
    """A value described with a band word must fall inside that band ("moderate ... AQI 135" is false)."""
    errors: List[str] = []
    for check in checks:
        bands, lo = [], float("-inf")
        for b in get_knowledge().phrasing[check["bands"]]:      # band = (previous max, max]
            bands.append((b["word"].lower(), lo, float(b["max"])))
            lo = float(b["max"])
        # longest words first, so "unhealthy for sensitive groups" is not also read as "unhealthy"
        word_rx = re.compile(r"\b(" + "|".join(re.escape(w) for w, _, _ in sorted(bands, key=lambda b: -len(b[0]))) + r")\b")
        span = {w: (a, b) for w, a, b in bands}
        win = int(check.get("window_chars", 40))
        for m in re.finditer(check["value_regex"], text):
            value = float(m.group(1))
            start, end = m.start(), m.end()
            sent_start = max(text.rfind(c, 0, start) for c in ".;\n") + 1
            sent_end = min([i for i in (text.find(c, end) for c in ".;\n") if i != -1] or [len(text)])
            window = text[max(sent_start, start - win):min(sent_end, end + win)]
            for w in {x.group(1) for x in word_rx.finditer(window)}:
                a, b = span[w]
                if not a < value <= b:
                    errors.append(f"calls {check['label']} {m.group(1)} '{w}' (that band is above {a:g} up to {b:g})")
    return errors


def verify(answer: str, sheet: str, ledger_ids: Set[str], known_names: List[str],
           must_mention: Optional[str] = None, recommended: bool = True,
           ledger_values: Optional[Dict[str, object]] = None) -> GroundingResult:
    cfg = get_knowledge().grounding
    violations: List[str] = []
    checked = 0
    low_answer = answer.lower()

    cites = _CITE.findall(answer)
    bad = sorted({c for c in cites if c not in ledger_ids})
    if bad:
        violations.append("unknown citation ids " + ", ".join(bad))
    if not cites:
        violations.append("no citations")

    # citation ↔ value consistency
    line_nums: Dict[str, Set[str]] = {}
    for ln in sheet.splitlines():
        ids_here = _CITE.findall(ln)
        nums_here: Set[str] = set()
        for m in _NUM.finditer(_CITE.sub(" ", ln)):
            nums_here |= _variants(m.group(1))
        for i in ids_here:
            line_nums.setdefault(i, set()).update(nums_here)
    win = int(cfg.get("citation_window_chars", 14))
    for m in _CITE.finditer(answer):
        cid = m.group(1)
        if cid not in line_nums or not line_nums[cid]:
            continue
        if ledger_values is not None and not _numeric(ledger_values.get(cid)):
            continue                                  # names, windows etc. are checked by the name/time rules
        before = _CITE.sub(" ", answer[max(0, m.start() - win - 12):m.start()])
        after = _CITE.sub(" ", answer[m.end():m.end() + win + 12])
        near = [x.group(1) for x in _NUM.finditer(before) if x.end() >= len(before) - win][-1:] + \
               [x.group(1) for x in _NUM.finditer(after) if x.start() <= win][:1]
        if near and not any(n in line_nums[cid] for n in near):
            violations.append(f"[{cid}] cited next to {near[0]}, which is not that fact's value")

    body = _CITE.sub(" ", answer)
    sheet_times = {f"{int(h):02d}:{m}" for h, m in _TIME.findall(sheet)}
    for h, m in _TIME.findall(body):
        checked += 1
        if f"{int(h):02d}:{m}" not in sheet_times:
            violations.append(f"time {h}:{m} not in fact sheet")
    for h, m, ap in _AMPM.findall(body):
        checked += 1
        hh = int(h) % 12 + (12 if ap.lower().startswith("p") else 0)
        if f"{hh:02d}:{m or '00'}" not in sheet_times:
            violations.append(f"time {h}{':' + m if m else ''} {ap} not in fact sheet")
    body = _AMPM.sub(" ", _TIME.sub(" ", body))

    sheet_q = _quantities(sheet)
    rx, fam = _unit_regex()
    for m in rx.finditer(body):
        checked += 1
        if (m.group(1), fam[m.group(2).lower()]) not in sheet_q:
            violations.append(f"'{m.group(0).strip()}' not in fact sheet")
    bare = rx.sub(" ", body)
    sheet_nums: Set[str] = set()
    for m in _NUM.finditer(sheet):
        sheet_nums |= _variants(m.group(1))
    for m in _NUM.finditer(bare):
        checked += 1
        if m.group(1) not in sheet_nums:
            violations.append(f"number {m.group(1)} not in fact sheet")

    sheet_lower = sheet.lower()
    known_terms = {t.lower() for t in cfg.get("known_terms", [])}
    names_lower = [n.lower() for n in known_names if n]
    for m in _NAME.finditer(_CITE.sub(" ", answer)):
        name = m.group(1)
        low = name.lower()
        if low in known_terms or low in sheet_lower:
            continue
        checked += 1
        if not any(fuzz.partial_ratio(low, n) >= cfg["name_match_ratio"] for n in names_lower):
            violations.append(f"place/name '{name}' not in fact sheet")

    if must_mention and fuzz.partial_ratio(must_mention.lower(), low_answer) < cfg["mention_match_ratio"]:
        violations.append(f"place '{must_mention}' not mentioned")
    if not recommended:
        nr = cfg["not_recommended"]
        if not any(p in low_answer for p in nr["required_any"]):
            violations.append("result is not recommended but the answer doesn't say so")
        for p in nr["forbidden"]:
            if p in low_answer:
                violations.append(f"'{p}' used for a not-recommended result")

    if "HEALTH WARNING:" in sheet and not any(w in low_answer for w in cfg["health_warning"]["required_any"]):
        violations.append("health warning in the fact sheet is not stated in the answer")
    has_options = any(ln.startswith("OTHER OPTIONS:") for ln in sheet.splitlines()) and \
        len(sheet.split("OTHER OPTIONS:", 1)[1].split("LIMITATIONS:", 1)[0].strip()) > 0
    if has_options:
        for phrase in cfg["absence_claims"]:
            if phrase in low_answer:
                violations.append(f"says '{phrase}' but the fact sheet lists other options")
    violations += _comparison_errors(_CITE.sub(" ", answer), cfg.get("comparisons"))
    violations += _time_comparison_errors(_CITE.sub(" ", answer), cfg.get("comparisons"))
    sheet_lines = [ln.lower() for ln in sheet.splitlines()]
    for rule in cfg["polarity"]:
        triggers = [t.lower() for t in rule["flag_triggers"]]
        flagged = any(any(t in ln for t in triggers) and any(w.lower() in ln for w in rule["flagged_when"])
                      for ln in sheet_lines)
        if not flagged:
            for rx in rule.get("forbidden_when_unflagged_regex", []):
                m = re.search(rx, low_answer)
                if m:
                    violations.append(f"says '{m.group(0)}' but the data does not flag {rule['flag_triggers'][0]}")
        if flagged:
            for phrase in rule["forbidden"]:
                if phrase in low_answer:
                    violations.append(f"says '{phrase}' but the data flags {rule['flag_triggers'][0]}")
            for rx in rule.get("forbidden_regex", []):
                m = re.search(rx, low_answer)
                if m:
                    violations.append(f"says '{m.group(0)}' but the data flags {rule['flag_triggers'][0]}")

    violations += _band_errors(_CITE.sub(" ", low_answer), cfg.get("band_checks") or [])

    # repetition: a small model looping on one phrase ("X; X; X; ...") is a failed answer even if X is true
    plain = re.sub(r"\s+", " ", _CITE.sub("", answer))
    min_len = int(cfg.get("repetition_min_chars", 40))
    for i in range(0, max(0, len(plain) - 2 * min_len), 10):
        chunk = plain[i:i + min_len]
        if plain.count(chunk) >= 2 and chunk.strip():
            violations.append(f"repeated text: '{chunk.strip()[:40]}…'")
            break

    return GroundingResult(status="failed" if violations else "passed", violations=violations[:12],
                           facts_checked=checked)
