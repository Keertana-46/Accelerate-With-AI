"""Pipeline state model shared across every medallion phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineState:
    """Mutable state carried through the medallion pipeline.

    The state stores identity fields, per-phase outputs, human-in-the-loop
    approval flags, cumulative rejection feedback and general bookkeeping.
    """

    # Identity
    run_id: str = ""
    business_intent: str = ""
    csv_paths: list[str] = field(default_factory=list)

    # Phase outputs
    profile_path: str = ""
    sttm_bronze_path: str = ""
    sttm_silver_path: str = ""
    sttm_gold_path: str = ""
    bronze_parquet_paths: list[str] = field(default_factory=list)
    silver_parquet_path: str = ""
    gold_parquet_paths: list[str] = field(default_factory=list)
    report_path: str = ""

    # Approval flags
    bronze_approved: bool = False
    silver_approved: bool = False
    gold_approved: bool = False

    # Cumulative rejection feedback
    bronze_feedback: list[str] = field(default_factory=list)
    silver_feedback: list[str] = field(default_factory=list)
    gold_feedback: list[str] = field(default_factory=list)

    # Bookkeeping
    current_phase: str = "idle"
    errors: list[str] = field(default_factory=list)
    scratchpad: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state into a plain dictionary."""
        return {
            "run_id": self.run_id,
            "business_intent": self.business_intent,
            "csv_paths": list(self.csv_paths),
            "profile_path": self.profile_path,
            "sttm_bronze_path": self.sttm_bronze_path,
            "sttm_silver_path": self.sttm_silver_path,
            "sttm_gold_path": self.sttm_gold_path,
            "bronze_parquet_paths": list(self.bronze_parquet_paths),
            "silver_parquet_path": self.silver_parquet_path,
            "gold_parquet_paths": list(self.gold_parquet_paths),
            "report_path": self.report_path,
            "bronze_approved": self.bronze_approved,
            "silver_approved": self.silver_approved,
            "gold_approved": self.gold_approved,
            "bronze_feedback": list(self.bronze_feedback),
            "silver_feedback": list(self.silver_feedback),
            "gold_feedback": list(self.gold_feedback),
            "current_phase": self.current_phase,
            "errors": list(self.errors),
            "scratchpad": dict(self.scratchpad),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineState":
        """Reconstruct a :class:`PipelineState` from a dictionary."""
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in (data or {}).items() if k in known}
        return cls(**filtered)
