"""Short-lived conversation memory for follow-ups ("what about Sunday?", "somewhere closer?")."""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from app.config import get_settings


class SessionStore:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    def get(self, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        item = self._data.get(session_id)
        if not item or time.monotonic() - item["ts"] > get_settings().SESSION_TTL_S:
            self._data.pop(session_id, None)
            return None
        return item

    def save(self, session_id: Optional[str], **values: Any) -> str:
        sid = session_id or str(uuid.uuid4())
        item = self._data.get(sid, {})
        item.update(values)
        item["ts"] = time.monotonic()
        self._data[sid] = item
        if len(self._data) > get_settings().SESSION_MAX:
            oldest = min(self._data, key=lambda k: self._data[k]["ts"])
            self._data.pop(oldest, None)
        return sid
