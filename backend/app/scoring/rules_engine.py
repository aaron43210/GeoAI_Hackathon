"""Three-valued rule evaluation over a dict of variables (knowledge/rules.yaml, knowledge/advice.yaml syntax).

TRUE: the condition holds on live data · FALSE: it doesn't · UNKNOWN: a needed variable is missing.
A rule that references a limit the activity/profile doesn't define is not applicable (FALSE).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Tri(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


_OPS = {
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b, "in": lambda a, b: a in b, "not_in": lambda a, b: a not in b,
}


@dataclass
class RuleHit:
    id: str
    decision: str
    severity: str
    state: Tri
    text: str
    var: Optional[str]
    value: Any
    limit: Any
    penalty: float = 0.0


def evaluate_condition(cond: Dict[str, Any], vars_: Dict[str, Any], limits: Dict[str, Any]
                       ) -> Tuple[Tri, Optional[str], Any, Any]:
    if "all" in cond or "any" in cond:
        parts = [evaluate_condition(c, vars_, limits) for c in cond.get("all", cond.get("any", []))]
        states = [p[0] for p in parts]
        lead = next((p for p in parts if p[1] not in ("is_day", "daylight_required")), parts[0])
        if "all" in cond:
            if Tri.FALSE in states:
                return Tri.FALSE, *lead[1:]
            return (Tri.UNKNOWN if Tri.UNKNOWN in states else Tri.TRUE), *lead[1:]
        if Tri.TRUE in states:
            return Tri.TRUE, *lead[1:]
        return (Tri.UNKNOWN if Tri.UNKNOWN in states else Tri.FALSE), *lead[1:]

    var = cond["var"]
    if "limit" in cond:
        if cond["limit"] not in limits:
            return Tri.FALSE, var, None, None
        threshold = limits[cond["limit"]]
    else:
        threshold = cond["value"]
    value = vars_.get(var)
    if value is None:
        return Tri.UNKNOWN, var, None, threshold
    return (Tri.TRUE if _OPS[cond["op"]](value, threshold) else Tri.FALSE), var, value, threshold


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return str(v)


def evaluate_rules(rules: List[Dict[str, Any]], vars_: Dict[str, Any], limits: Dict[str, Any]) -> List[RuleHit]:
    hits = []
    for rule in rules:
        state, var, value, limit = evaluate_condition(rule["when"], vars_, limits)
        if state == Tri.FALSE:
            continue
        text = rule["text"].format(value=_fmt(value) if value is not None else "?",
                                   limit=_fmt(limit) if limit is not None else "?")
        hits.append(RuleHit(id=rule["id"], decision=rule["decision"], severity=rule["severity"], state=state,
                            text=text, var=var, value=value, limit=limit, penalty=float(rule.get("penalty", 0))))
    # a caution is redundant when the matching avoid rule already fired
    avoid_vars = {h.var for h in hits if h.decision == "avoid" and h.state == Tri.TRUE}
    hits = [h for h in hits if not (h.decision == "caution" and h.var in avoid_vars)]
    # among cautions on the same variable keep only the strongest (highest penalty)
    best: Dict[str, RuleHit] = {}
    out = []
    for h in hits:
        if h.decision == "caution" and h.state == Tri.TRUE and h.var:
            if h.var not in best or h.penalty > best[h.var].penalty:
                best[h.var] = h
        else:
            out.append(h)
    return out + list(best.values())
