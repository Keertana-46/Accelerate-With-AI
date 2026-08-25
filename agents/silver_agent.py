"""Silver refinement agent: cleansing, derived fields and deduplication."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core import config
from core.audit import AuditLogger
from core.observability import AgentTrace


def _find_column(columns: list[str], candidates: list[str]) -> str:
    """Return the first column whose lowercase name matches a candidate."""
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for candidate in candidates:
        for low, original in lowered.items():
            if candidate in low:
                return original
    return ""


def _normalize_dates(series: pd.Series) -> pd.Series:
    """Normalize mixed date formats to ``YYYY-MM-DD`` strings."""
    parsed = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=False)
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), "")


def run_silver_agent(bronze_parquet_paths: list[str], sttm_silver_path: str,
                     run_id: str) -> str:
    """Combine bronze frames, add derived fields and return the silver path."""
    if not bronze_parquet_paths:
        raise ValueError("bronze_parquet_paths must not be empty")
    if not run_id:
        raise ValueError("run_id must be provided")

    audit = AuditLogger(run_id)
    trace = AgentTrace("silver_agent", run_id)
    trace.set_input({"bronze_paths": bronze_parquet_paths, "sttm": sttm_silver_path})
    audit.log("silver_agent", "phase_3", "start", {"inputs": len(bronze_parquet_paths)})

    try:
        frames = []
        for path in bronze_parquet_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"bronze parquet not found: {path}")
            frames.append(pd.read_parquet(path))
        combined = pd.concat(frames, ignore_index=True)
        trace.add_plan("concatenated bronze frames")

        columns = list(combined.columns)

        date_like = [c for c in columns if "date" in c.lower()]
        for col in date_like:
            combined[col] = _normalize_dates(combined[col])
            trace.add_tool_call("normalize_dates", {"column": col})

        unit_price = _find_column(columns, ["unit_price", "price", "unitprice"])
        quantity = _find_column(columns, ["quantity", "qty"])
        total = _find_column(columns, ["total_amount", "total", "amount"])

        if unit_price and quantity:
            combined["total_amount_calculated"] = (
                pd.to_numeric(combined[unit_price], errors="coerce")
                * pd.to_numeric(combined[quantity], errors="coerce")
            )
            trace.add_verification("derived total_amount_calculated")
        else:
            combined["total_amount_calculated"] = pd.NA

        if total and "total_amount_calculated" in combined:
            stated = pd.to_numeric(combined[total], errors="coerce")
            calc = pd.to_numeric(combined["total_amount_calculated"], errors="coerce")
            variance = (stated - calc).abs()
            combined["amount_variance_flag"] = (variance > 0.01).map(
                {True: "variance", False: "ok"}
            )
        else:
            combined["amount_variance_flag"] = "unknown"

        order_date = _find_column(columns, ["order_date", "orderdate"])
        ship_date = _find_column(columns, ["ship_date", "shipdate", "shipped_date"])
        if order_date and ship_date:
            start = pd.to_datetime(combined[order_date], errors="coerce")
            end = pd.to_datetime(combined[ship_date], errors="coerce")
            combined["days_to_ship"] = (end - start).dt.days
            trace.add_verification("derived days_to_ship")
        else:
            combined["days_to_ship"] = pd.NA

        order_id = _find_column(columns, ["order_id", "orderid"])
        if order_id:
            before = len(combined)
            combined = combined.drop_duplicates(subset=[order_id], keep="first")
            trace.add_verification(
                f"deduplicated on {order_id}: {before - len(combined)} rows removed"
            )

        combined = combined.reset_index(drop=True)
        combined["_run_id"] = run_id
        combined["_refined_at"] = datetime.now(timezone.utc).isoformat()

        out_path = Path(config.SILVER_DIR) / f"silver_{run_id}.parquet"
        combined.to_parquet(out_path, index=False)

        audit.log("silver_agent", "phase_3", "complete", {"silver_path": str(out_path)})
        trace.finish(output={"silver_path": str(out_path), "rows": len(combined)})
        return str(out_path)
    except Exception as exc:
        audit.log("silver_agent", "phase_3", "error", {"error": str(exc)})
        trace.finish(output={"error": str(exc)}, status="error")
        raise
