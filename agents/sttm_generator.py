"""Source-to-target mapping (STTM) generation for each medallion layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from agents._llm import LLMError, call_llm, extract_json
from core import config
from core.audit import AuditLogger
from core.observability import AgentTrace

STTM_COLUMNS = [
    "source_column",
    "target_column",
    "target_type",
    "transformation",
    "description",
]

_VALID_LAYERS = {"bronze", "silver", "gold"}


def _sttm_path(layer: str, run_id: str) -> Path:
    """Return the output CSV path for ``layer`` and ``run_id``."""
    folder = {
        "bronze": config.STTM_BRONZE,
        "silver": config.STTM_SILVER,
        "gold": config.STTM_GOLD,
    }[layer]
    return Path(folder) / f"sttm_{layer}_{run_id}.csv"


def _normalize_name(name: str) -> str:
    """Normalize a column name to snake_case ASCII."""
    text = "".join(ch if (ch.isalnum() or ch == " ") else " " for ch in str(name))
    return "_".join(part.lower() for part in text.split() if part) or "column"


def _write_sttm(rows: list[dict[str, str]], path: Path) -> None:
    """Write STTM ``rows`` to ``path`` using the canonical column order."""
    frame = pd.DataFrame(rows, columns=STTM_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _load_profile(profile_path: str) -> dict[str, Any]:
    """Load a profile JSON produced by the profiler agent."""
    path = Path(profile_path)
    if not path.exists():
        raise FileNotFoundError(f"profile not found: {profile_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _bronze_rule(column: str, info: dict[str, Any]) -> str:
    """Choose a deterministic bronze transformation rule for a column."""
    flags = info.get("quality_flags", [])
    if "currency_formatted" in flags:
        return "strip_currency"
    if info.get("is_numeric_like"):
        return "cast_numeric"
    if len(info.get("date_formats", [])) >= 1:
        return "to_date"
    return "passthrough"


def _bronze_type(rule: str) -> str:
    """Map a bronze rule to a target data type."""
    return {
        "strip_currency": "float",
        "cast_numeric": "float",
        "to_date": "string",
        "title_case": "string",
        "passthrough": "string",
    }.get(rule, "string")


def _fallback_bronze_rows(profile: dict[str, Any]) -> list[dict[str, str]]:
    """Build deterministic bronze STTM rows directly from the profile."""
    stats = profile.get("statistics", {}).get("columns", {})
    meanings = profile.get("semantic_meanings", {})
    rows: list[dict[str, str]] = []
    for column, info in stats.items():
        rule = _bronze_rule(column, info)
        rows.append(
            {
                "source_column": column,
                "target_column": _normalize_name(column),
                "target_type": _bronze_type(rule),
                "transformation": rule,
                "description": meanings.get(column, f"Bronze mapping for {column}"),
            }
        )
    return rows


def _llm_bronze_rows(profile: dict[str, Any], business_intent: str,
                     reviewer_feedback: str) -> list[dict[str, str]]:
    """Ask the LLM to produce bronze STTM rows, falling back deterministically."""
    stats = profile.get("statistics", {}).get("columns", {})
    columns_payload = {
        col: {
            "quality_flags": info.get("quality_flags", []),
            "is_numeric_like": info.get("is_numeric_like", False),
            "date_formats": info.get("date_formats", []),
            "sample": info.get("sample", []),
        }
        for col, info in stats.items()
    }
    system = (
        "You are a data engineer building a Bronze source-to-target mapping. "
        "Return a JSON array. Each element must be an object with keys: "
        "source_column, target_column, target_type, transformation, description. "
        "Allowed transformation values: strip_currency, cast_numeric, to_date, "
        "title_case, passthrough. Respond with JSON only."
    )
    user = json.dumps(
        {
            "business_intent": business_intent,
            "reviewer_feedback": reviewer_feedback,
            "columns": columns_payload,
        }
    )
    try:
        raw = call_llm(system, user, max_tokens=2048)
        parsed = extract_json(raw)
    except (LLMError, ValueError):
        return _fallback_bronze_rows(profile)

    if not isinstance(parsed, list) or not parsed:
        return _fallback_bronze_rows(profile)

    valid_source = set(stats.keys())
    allowed = {"strip_currency", "cast_numeric", "to_date", "title_case", "passthrough"}
    rows: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source_column", "")).strip()
        if source not in valid_source:
            continue
        rule = str(item.get("transformation", "passthrough")).strip()
        if rule not in allowed:
            rule = _bronze_rule(source, stats[source])
        rows.append(
            {
                "source_column": source,
                "target_column": _normalize_name(
                    item.get("target_column") or source
                ),
                "target_type": str(item.get("target_type") or _bronze_type(rule)),
                "transformation": rule,
                "description": str(item.get("description") or f"Bronze mapping for {source}"),
            }
        )
    return rows or _fallback_bronze_rows(profile)


def _silver_rows_from_bronze(bronze_sttm_path: str) -> list[dict[str, str]]:
    """Derive silver STTM rows programmatically from a bronze STTM."""
    path = Path(bronze_sttm_path)
    if not path.exists():
        raise FileNotFoundError(f"bronze STTM not found: {bronze_sttm_path}")
    bronze = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows: list[dict[str, str]] = []
    targets = set()
    for _, row in bronze.iterrows():
        target = row["target_column"]
        targets.add(target)
        rows.append(
            {
                "source_column": target,
                "target_column": target,
                "target_type": row.get("target_type", "string"),
                "transformation": "passthrough",
                "description": f"Silver cleansed passthrough for {target}",
            }
        )

    derived = [
        ("total_amount_calculated", "float", "unit_price * quantity when available"),
        ("amount_variance_flag", "string", "flag when stated total differs from calculated"),
        ("days_to_ship", "float", "difference between ship and order dates"),
    ]
    for name, dtype, desc in derived:
        if name not in targets:
            rows.append(
                {
                    "source_column": "",
                    "target_column": name,
                    "target_type": dtype,
                    "transformation": "derive",
                    "description": desc,
                }
            )
    return rows


def _gold_rows_from_silver(silver_parquet_path: str) -> list[dict[str, str]]:
    """Derive gold STTM rows from a silver parquet schema without hardcoding.

    Target tables are declared as rows whose ``transformation`` is ``target_table``
    so downstream agents can discover them dynamically.
    """
    path = Path(silver_parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"silver parquet not found: {silver_parquet_path}")
    frame = pd.read_parquet(path)
    columns = list(frame.columns)
    numeric_cols = [
        c for c in columns
        if pd.api.types.is_numeric_dtype(frame[c]) and not c.startswith("_")
    ]
    text_cols = [
        c for c in columns
        if not pd.api.types.is_numeric_dtype(frame[c]) and not c.startswith("_")
    ]
    date_cols = [c for c in columns if "date" in c.lower()]
    id_cols = [c for c in columns if c.lower().endswith("_id") or c.lower() == "id"
               or "customer" in c.lower()]

    rows: list[dict[str, str]] = []

    for col in numeric_cols[:3]:
        for dim in text_cols[:3]:
            rows.append(
                {
                    "source_column": f"{dim}|{col}",
                    "target_column": f"revenue_by_{dim}",
                    "target_type": "table",
                    "transformation": "target_table",
                    "description": f"Aggregate {col} grouped by {dim}",
                }
            )

    if date_cols and numeric_cols:
        rows.append(
            {
                "source_column": f"{date_cols[0]}|{numeric_cols[0]}",
                "target_column": "monthly_trend",
                "target_type": "table",
                "transformation": "target_table",
                "description": f"Monthly trend of {numeric_cols[0]} by {date_cols[0]}",
            }
        )

    if id_cols and numeric_cols:
        rows.append(
            {
                "source_column": f"{id_cols[0]}|{numeric_cols[0]}",
                "target_column": "top_customers",
                "target_type": "table",
                "transformation": "target_table",
                "description": f"Top entities by {numeric_cols[0]} for {id_cols[0]}",
            }
        )

    if not rows:
        rows.append(
            {
                "source_column": "",
                "target_column": "summary",
                "target_type": "table",
                "transformation": "target_table",
                "description": "Overall row count summary",
            }
        )
    return rows


def run_sttm_generator(
    profile_path: str,
    layer: str,
    business_intent: str,
    run_id: str,
    prev_sttm_path: str = "",
    reviewer_feedback: str = "",
    silver_parquet_path: str = "",
) -> str:
    """Generate an STTM for ``layer`` and return the written CSV path."""
    if layer not in _VALID_LAYERS:
        raise ValueError(f"layer must be one of {_VALID_LAYERS}, got {layer}")
    if not run_id:
        raise ValueError("run_id must be provided")

    audit = AuditLogger(run_id)
    trace = AgentTrace(f"sttm_{layer}", run_id)
    trace.set_input(
        {
            "layer": layer,
            "profile_path": profile_path,
            "reviewer_feedback": reviewer_feedback,
        }
    )
    audit.log("sttm_generator", f"sttm_{layer}", "start", {"layer": layer})

    out_path = _sttm_path(layer, run_id)
    revision = bool(reviewer_feedback) and out_path.exists()
    if revision:
        trace.add_plan("revision mode: regenerate with cumulative feedback")

    try:
        if layer == "bronze":
            profile = _load_profile(profile_path)
            trace.add_plan("generate bronze STTM via LLM")
            rows = _llm_bronze_rows(profile, business_intent, reviewer_feedback)
        elif layer == "silver":
            source = prev_sttm_path or str(_sttm_path("bronze", run_id))
            trace.add_plan("derive silver STTM from bronze STTM")
            rows = _silver_rows_from_bronze(source)
        else:
            trace.add_plan("derive gold STTM from silver parquet schema")
            rows = _gold_rows_from_silver(silver_parquet_path)

        _write_sttm(rows, out_path)
        trace.add_verification(f"wrote {len(rows)} STTM rows")
        audit.log(
            "sttm_generator",
            f"sttm_{layer}",
            "complete",
            {"sttm_path": str(out_path), "rows": len(rows), "revision": revision},
        )
        trace.finish(output={"sttm_path": str(out_path), "rows": len(rows)})
        return str(out_path)
    except Exception as exc:
        audit.log("sttm_generator", f"sttm_{layer}", "error", {"error": str(exc)})
        trace.finish(output={"error": str(exc)}, status="error")
        raise
