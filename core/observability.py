"""Lightweight agent tracing for observability."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import TRACES_DIR


class AgentTrace:
    """Capture the execution of a single agent for later inspection.

    The trace records the input, planning steps, tool calls, verification
    steps, output, status and duration, and persists to a JSON file named
    ``trace_<agent>_<run_id>.json``.
    """

    def __init__(self, agent: str, run_id: str) -> None:
        """Create a trace for ``agent`` within ``run_id``."""
        if not agent or not run_id:
            raise ValueError("agent and run_id must be non-empty strings")
        self.agent = agent
        self.run_id = run_id
        self.input: Any = None
        self.plan_steps: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.verification_steps: list[str] = []
        self.output: Any = None
        self.status: str = "started"
        self.duration: float = 0.0
        self._start = time.perf_counter()
        self.path = Path(TRACES_DIR) / f"trace_{agent}_{run_id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def set_input(self, value: Any) -> None:
        """Record the agent input."""
        self.input = value

    def add_plan(self, step: str) -> None:
        """Record a planning step."""
        self.plan_steps.append(step)

    def add_tool_call(self, name: str, args: dict[str, Any] | None = None,
                      result: Any = None) -> None:
        """Record a tool invocation with its arguments and result."""
        self.tool_calls.append({"name": name, "args": args or {}, "result": result})

    def add_verification(self, step: str) -> None:
        """Record a verification step."""
        self.verification_steps.append(step)

    def finish(self, output: Any = None, status: str = "success") -> dict[str, Any]:
        """Finalize the trace, persist it and return the record."""
        self.output = output
        self.status = status
        self.duration = round(time.perf_counter() - self._start, 6)
        record = self.to_dict()
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=True, indent=2, default=str)
        return record

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace to a dictionary."""
        return {
            "agent": self.agent,
            "run_id": self.run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "input": self.input,
            "plan_steps": list(self.plan_steps),
            "tool_calls": list(self.tool_calls),
            "verification_steps": list(self.verification_steps),
            "output": self.output,
            "status": self.status,
            "duration": self.duration,
        }
