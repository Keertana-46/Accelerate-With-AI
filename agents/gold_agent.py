"""Gold aggregation agent producing analytics-ready tables dynamically."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core import config
from core.audit import AuditLogger
from core.observability import AgentTrace


def _resolve_amount_column(frame: pd.DataFrame) -> str:
    """Resolve the best available monetary column, deriving one if needed.

    Preference order: ``total_amount`` -> ``total_amount_calculated`` ->
    a revenue/amount-like column -> derived ``unit_price * quantity``.
    """
    lowered = {c.lower(): c for c in frame.columns}

    for name in ("total_amount", "total_amount_calculated"):
        if name in lowered and frame[lowered[name]].notna().any():
            return lowered[name]

    for low, original in lowered.items():
        if low.startswith("_"):
            continue
        if any(tok in low for tok in ("revenue", "amount", "sales", "total")):
            if pd.api.types.is_numeric_dtype(frame[original]):
                return original

    price = next((lowered[k] for k in lowered if "price" in k), "")
    qty = next((lowered[k] for k in lowered if "quantity" in k or k == "qty"), "")
    if price and qty:
        frame["_derived_amount"] = (
            pd.to_numeric(frame[price], errors="coerce")
            * pd.to_numeric(frame[qty], errors="coerce")
        )
        return "_derived_amount"

    raise ValueError("could not resolve an amount column for gold aggregation")


def _read_targets(sttm_gold_path: str) -> pd.DataFrame:
    """Read the gold STTM and return only target-table rows."""
    path = Path(sttm_gold_path)
    if not path.exists():
        raise FileNotFoundError(f"gold STTM not found: {sttm_gold_path}")
    sttm = pd.read_csv(path, dtype=str, keep_default_na=False)
    return sttm[sttm["transformation"] == "target_table"]


def _write_table(frame: pd.DataFrame, table_name: str, run_id: str) -> str:
    """Attach run metadata and write a gold table parquet file."""
    frame = frame.copy()
    frame["_run_id"] = run_id
    out_path = Path(config.GOLD_DIR) / f"gold_{table_name}_{run_id}.parquet"
    frame.to_parquet(out_path, index=False)
    return str(out_path)


def run_gold_agent(silver_parquet_paths: list[str], sttm_gold_path: str,
                   run_id: str) -> list[str]:
    """Build gold analytics tables from the silver layer and gold STTM."""
    if not silver_parquet_paths:
        raise ValueError("silver_parquet_paths must not be empty")
    if not run_id:
        raise ValueError("run_id must be provided")

    audit = AuditLogger(run_id)
    trace = AgentTrace("gold_agent", run_id)
    trace.set_input({"silver_paths": silver_parquet_paths, "sttm": sttm_gold_path})
    audit.log("gold_agent", "phase_4", "start", {"inputs": len(silver_parquet_paths)})

    try:
        frames = []
        for path in silver_parquet_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"silver parquet not found: {path}")
            frames.append(pd.read_parquet(path))
        silver = pd.concat(frames, ignore_index=True)

        amount_col = _resolve_amount_column(silver)
        trace.add_verification(f"resolved amount column: {amount_col}")
        silver[amount_col] = pd.to_numeric(silver[amount_col], errors="coerce")

        targets = _read_targets(sttm_gold_path)
        outputs: list[str] = []

        for _, row in targets.iterrows():
            table_name = row["target_column"]
            source = row.get("source_column", "")
            trace.add_plan(f"build table {table_name}")

            if table_name.startswith("revenue_by_") and "|" in source:
                dim = source.split("|", 1)[0]
                if dim not in silver.columns:
                    continue
                grouped = (
                    silver.groupby(dim, dropna=False)[amount_col]
                    .sum()
                    .reset_index()
                    .rename(columns={amount_col: "total_revenue"})
                    .sort_values("total_revenue", ascending=False)
                )
                outputs.append(_write_table(grouped, table_name, run_id))

            elif table_name == "monthly_trend" and "|" in source:
                date_col = source.split("|", 1)[0]
                if date_col not in silver.columns:
                    continue
                dates = pd.to_datetime(silver[date_col], errors="coerce")
                monthly = silver.assign(_month=dates.dt.strftime("%Y-%m"))
                trend = (
                    monthly.dropna(subset=["_month"])
                    .groupby("_month")[amount_col]
                    .sum()
                    .reset_index()
                    .rename(columns={amount_col: "total_revenue", "_month": "month"})
                    .sort_values("month")
                )
                outputs.append(_write_table(trend, table_name, run_id))

            elif table_name == "top_customers" and "|" in source:
                id_col = source.split("|", 1)[0]
                if id_col not in silver.columns:
                    continue
                top = (
                    silver.groupby(id_col, dropna=False)[amount_col]
                    .sum()
                    .reset_index()
                    .rename(columns={amount_col: "total_revenue"})
                    .sort_values("total_revenue", ascending=False)
                    .head(10)
                )
                outputs.append(_write_table(top, table_name, run_id))

            else:
                summary = pd.DataFrame(
                    {
                        "metric": ["row_count", "total_revenue"],
                        "value": [len(silver), float(silver[amount_col].sum())],
                    }
                )
                outputs.append(_write_table(summary, table_name, run_id))

        if not outputs:
            summary = pd.DataFrame(
                {
                    "metric": ["row_count", "total_revenue"],
                    "value": [len(silver), float(silver[amount_col].sum())],
                }
            )
            outputs.append(_write_table(summary, "summary", run_id))

        audit.log("gold_agent", "phase_4", "complete", {"outputs": outputs})
        trace.finish(output={"outputs": outputs})
        return outputs
    except Exception as exc:
        audit.log("gold_agent", "phase_4", "error", {"error": str(exc)})
        trace.finish(output={"error": str(exc)}, status="error")
        raise
