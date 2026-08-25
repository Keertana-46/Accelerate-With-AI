"""Bronze ingestion agent applying row-level STTM rules."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core import config
from core.audit import AuditLogger
from core.observability import AgentTrace

# Currency symbols kept ASCII-safe in source via code points (pound, euro).
_CURRENCY_STRIP_CLASS = "[,$" + chr(0xA3) + chr(0x20AC) + r"\s]"


def _load_sttm(sttm_path: str) -> pd.DataFrame:
    """Load a bronze STTM CSV, validating required columns."""
    path = Path(sttm_path)
    if not path.exists():
        raise FileNotFoundError(f"bronze STTM not found: {sttm_path}")
    sttm = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"source_column", "target_column", "transformation"}
    if not required.issubset(sttm.columns):
        raise ValueError(f"bronze STTM missing columns: {required - set(sttm.columns)}")
    return sttm


def _apply_rule(series: pd.Series, rule: str) -> pd.Series:
    """Apply a single bronze transformation rule to a column."""
    if rule == "strip_currency":
        cleaned = series.astype(str).str.replace(_CURRENCY_STRIP_CLASS, "", regex=True)
        return pd.to_numeric(cleaned, errors="coerce")
    if rule == "cast_numeric":
        cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
        return pd.to_numeric(cleaned, errors="coerce")
    if rule == "title_case":
        return series.astype(str).str.strip().str.title()
    if rule == "to_date":
        return series.astype(str).str.strip()
    return series.astype(str).str.strip()


def run_bronze_agent(csv_paths: list[str], sttm_bronze_path: str,
                     run_id: str) -> list[str]:
    """Ingest each CSV under STTM rules and return the parquet paths written."""
    if not csv_paths:
        raise ValueError("csv_paths must not be empty")
    if not run_id:
        raise ValueError("run_id must be provided")

    audit = AuditLogger(run_id)
    trace = AgentTrace("bronze_agent", run_id)
    trace.set_input({"csv_paths": csv_paths, "sttm": sttm_bronze_path})
    audit.log("bronze_agent", "phase_2", "start", {"files": len(csv_paths)})

    try:
        sttm = _load_sttm(sttm_bronze_path)
        outputs: list[str] = []

        for csv_path in csv_paths:
            source = Path(csv_path)
            if not source.exists():
                raise FileNotFoundError(f"source CSV not found: {csv_path}")
            frame = pd.read_csv(source, dtype=str, keep_default_na=False)
            result = pd.DataFrame(index=frame.index)

            for _, row in sttm.iterrows():
                src = row["source_column"]
                target = row["target_column"]
                rule = row["transformation"]
                if src not in frame.columns:
                    continue
                result[target] = _apply_rule(frame[src], rule)
                trace.add_tool_call("apply_rule", {"column": src, "rule": rule})

            result["_source_file"] = source.name
            result["_ingested_at"] = datetime.now(timezone.utc).isoformat()
            result["_run_id"] = run_id

            out_path = Path(config.BRONZE_DIR) / f"{source.stem}_{run_id}.parquet"
            result.to_parquet(out_path, index=False)
            outputs.append(str(out_path))
            trace.add_verification(f"wrote {len(result)} rows to {out_path.name}")

        audit.log("bronze_agent", "phase_2", "complete", {"outputs": outputs})
        trace.finish(output={"outputs": outputs})
        return outputs
    except Exception as exc:
        audit.log("bronze_agent", "phase_2", "error", {"error": str(exc)})
        trace.finish(output={"error": str(exc)}, status="error")
        raise
