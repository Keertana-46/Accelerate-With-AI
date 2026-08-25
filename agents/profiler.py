"""Data profiling agent with deterministic tools and optional LLM semantics."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from agents._llm import LLMError, call_llm, extract_json
from core import config
from core.audit import AuditLogger
from core.observability import AgentTrace

# Currency symbols kept ASCII-safe in source via code points (pound, euro).
_CURRENCY_SYMBOLS = chr(0xA3) + chr(0x20AC)
_CURRENCY_STRIP_CLASS = "[,$" + _CURRENCY_SYMBOLS + "]"
_CURRENCY_RE = re.compile(r"^\s*[-+]?[$\u00a3\u20ac]\s*[\d,]+(?:\.\d+)?\s*$")
_DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{2}/\d{2}/\d{4}$"),
    re.compile(r"^\d{2}-\d{2}-\d{4}$"),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
    re.compile(r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$"),
]


def _read_csv(file_path: str) -> pd.DataFrame:
    """Read a CSV file into a dataframe with basic validation."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"file has no rows: {file_path}")
    return frame


def inspect_files(file_path: str) -> dict[str, Any]:
    """Return shape, columns, dtypes, sample values and missing counts."""
    frame = _read_csv(file_path)
    sample: dict[str, list[str]] = {}
    missing: dict[str, int] = {}
    for column in frame.columns:
        series = frame[column]
        sample[column] = series.head(5).tolist()
        missing[column] = int((series.str.strip() == "").sum())
    return {
        "file_path": file_path,
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": list(frame.columns),
        "dtypes": {c: "string" for c in frame.columns},
        "sample_values": sample,
        "missing_counts": missing,
    }


def _looks_currency(values: list[str]) -> bool:
    """Return ``True`` when non-empty values look currency formatted."""
    non_empty = [v for v in values if v and v.strip()]
    if not non_empty:
        return False
    hits = sum(1 for v in non_empty if _CURRENCY_RE.match(v))
    return hits >= max(1, int(0.6 * len(non_empty)))


def _date_formats(values: list[str]) -> set[str]:
    """Return the set of date pattern signatures observed in ``values``."""
    signatures: set[str] = set()
    for value in values:
        value = value.strip()
        if not value:
            continue
        for pattern in _DATE_PATTERNS:
            if pattern.match(value):
                signatures.add(pattern.pattern)
                break
    return signatures


def profiler_tool(file_path: str) -> dict[str, Any]:
    """Compute per-column statistics and data-quality flags."""
    frame = _read_csv(file_path)
    columns: dict[str, Any] = {}
    for column in frame.columns:
        series = frame[column]
        values = series.tolist()
        non_empty = [v for v in values if v and v.strip()]
        flags: list[str] = []

        if _looks_currency(values):
            flags.append("currency_formatted")

        formats = _date_formats(values)
        if len(formats) > 1:
            flags.append("mixed_date_formats")

        numeric = pd.to_numeric(
            series.str.replace(_CURRENCY_STRIP_CLASS, "", regex=True).str.strip(),
            errors="coerce",
        )
        is_numeric = numeric.notna().sum() >= max(1, int(0.8 * len(non_empty))) \
            if non_empty else False

        columns[column] = {
            "count": len(values),
            "non_null": len(non_empty),
            "missing": len(values) - len(non_empty),
            "unique": int(series.nunique()),
            "is_numeric_like": bool(is_numeric),
            "date_formats": sorted(formats),
            "quality_flags": flags,
            "sample": non_empty[:5],
        }
    return {"file_path": file_path, "columns": columns}


def _quality_notes(stats: dict[str, Any]) -> list[str]:
    """Build human-readable quality notes from profiler statistics."""
    notes: list[str] = []
    for column, info in stats["columns"].items():
        if "currency_formatted" in info["quality_flags"]:
            notes.append(f"Column '{column}' contains currency formatted values.")
        if "mixed_date_formats" in info["quality_flags"]:
            notes.append(f"Column '{column}' has mixed date formats.")
        if info["missing"] > 0:
            notes.append(f"Column '{column}' has {info['missing']} missing values.")
    if not notes:
        notes.append("No structural quality issues detected.")
    return notes


def _semantic_meanings(columns: list[str], business_intent: str) -> dict[str, str]:
    """Ask the LLM for concise semantic meanings per column.

    Falls back to an empty mapping when no LLM is configured or the call fails.
    """
    system = (
        "You are a data analyst. Given column names and a business intent, "
        "return a JSON object mapping each column name to a short semantic "
        "meaning. Respond with JSON only."
    )
    user = json.dumps({"columns": columns, "business_intent": business_intent})
    try:
        raw = call_llm(system, user, max_tokens=1024)
        parsed = extract_json(raw)
    except (LLMError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items() if k in columns}


def run_profiler(file_path: str, business_intent: str, run_id: str) -> str:
    """Profile ``file_path`` and persist a JSON profile, returning its path."""
    if not run_id:
        raise ValueError("run_id must be provided")
    audit = AuditLogger(run_id)
    trace = AgentTrace("profiler", run_id)
    trace.set_input({"file_path": file_path, "business_intent": business_intent})
    audit.log("profiler", "phase_1", "start", {"file_path": file_path})

    try:
        trace.add_plan("inspect file structure")
        inspection = inspect_files(file_path)
        trace.add_tool_call("inspect_files", {"file_path": file_path}, inspection["shape"])

        trace.add_plan("compute column statistics")
        stats = profiler_tool(file_path)
        trace.add_tool_call("profiler_tool", {"file_path": file_path},
                            list(stats["columns"].keys()))

        trace.add_plan("derive semantic meanings via LLM")
        meanings = _semantic_meanings(inspection["columns"], business_intent)

        trace.add_plan("build quality notes")
        notes = _quality_notes(stats)
        trace.add_verification(f"generated {len(notes)} quality notes")

        profile = {
            "run_id": run_id,
            "file_path": file_path,
            "business_intent": business_intent,
            "generated_at": datetime.utcnow().isoformat(),
            "inspection": inspection,
            "statistics": stats,
            "semantic_meanings": meanings,
            "quality_notes": notes,
        }

        stem = Path(file_path).stem
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        out_path = Path(config.PROFILES_DIR) / f"profile_{stem}_{timestamp}.json"
        out_path.write_text(json.dumps(profile, ensure_ascii=True, indent=2),
                            encoding="utf-8")

        audit.log("profiler", "phase_1", "complete", {"profile_path": str(out_path)})
        trace.finish(output={"profile_path": str(out_path)})
        return str(out_path)
    except Exception as exc:
        audit.log("profiler", "phase_1", "error", {"error": str(exc)})
        trace.finish(output={"error": str(exc)}, status="error")
        raise
