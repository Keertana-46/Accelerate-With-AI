"""Append-only JSONL audit logging for pipeline runs."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import TRACES_DIR


class AuditLogger:
    """Thread-safe append-only audit logger.

    Every event is written as one JSON object per line to
    ``data/traces/audit_<run_id>.jsonl``.
    """

    def __init__(self, run_id: str) -> None:
        """Create an audit logger bound to ``run_id``."""
        if not run_id:
            raise ValueError("run_id must be a non-empty string")
        self.run_id = run_id
        self.path = Path(TRACES_DIR) / f"audit_{run_id}.jsonl"
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        agent: str,
        phase: str,
        event: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a single audit entry and return the written record."""
        entry = {
            "run_id": self.run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "phase": phase,
            "event": event,
            "details": details or {},
        }
        line = json.dumps(entry, ensure_ascii=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return entry

    def read_all(self) -> list[dict[str, Any]]:
        """Return every audit entry recorded for this run."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if raw:
                    records.append(json.loads(raw))
        return records
