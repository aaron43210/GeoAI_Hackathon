"""Evidence ledger: every value shown to the user gets an id (E1, E2, ...) with source, URL, time and method."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Evidence:
    id: str
    label: str
    value: Any
    unit: Optional[str]
    source: str
    url: Optional[str]
    kind: str                       # forecast | observation | osm | computed | modelled | heuristic | derived
    time: Optional[str] = None
    venue: Optional[str] = None
    method: Optional[str] = None
    retrieved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Ledger:
    items: List[Evidence] = field(default_factory=list)
    keys: Dict[str, str] = field(default_factory=dict)

    def add(self, label: str, value: Any, unit: Optional[str], source: str, url: Optional[str], kind: str,
            key: Optional[str] = None, **kw: Any) -> str:
        if key and key in self.keys:
            return self.keys[key]
        eid = f"E{len(self.items) + 1}"
        self.items.append(Evidence(id=eid, label=label, value=value, unit=unit, source=source, url=url, kind=kind, **kw))
        if key:
            self.keys[key] = eid
        return eid

    def get(self, key: str) -> Optional[str]:
        return self.keys.get(key)

    def ids(self) -> set:
        return {e.id for e in self.items}

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.items]
