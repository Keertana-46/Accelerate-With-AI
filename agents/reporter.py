"""On-demand HTML report generation with Plotly charts."""

from __future__ import annotations

import html
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

from core import config
from core.audit import AuditLogger
from core.memory import VectorMemory
from core.observability import AgentTrace


def _table_name_from_path(path: str, run_id: str) -> str:
    """Recover the gold table name from its parquet file path."""
    stem = Path(path).stem
    prefix = "gold_"
    suffix = f"_{run_id}"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem or "table"


def _load_table(path: str) -> pd.DataFrame:
    """Load a parquet gold table through DuckDB."""
    query = "SELECT * FROM read_parquet(?)"
    return duckdb.execute(query, [path]).df()


def _build_chart(name: str, frame: pd.DataFrame) -> go.Figure | None:
    """Choose a chart type for ``frame`` using simple schema heuristics."""
    if frame.empty:
        return None
    numeric_cols = [c for c in frame.columns
                    if pd.api.types.is_numeric_dtype(frame[c]) and not c.startswith("_")]
    text_cols = [c for c in frame.columns
                 if not pd.api.types.is_numeric_dtype(frame[c]) and not c.startswith("_")]
    if not numeric_cols or not text_cols:
        return None

    x = text_cols[0]
    y = numeric_cols[0]
    figure = go.Figure()

    if "month" in x.lower() or "date" in x.lower():
        figure.add_trace(go.Scatter(x=frame[x], y=frame[y], mode="lines+markers"))
    elif name.startswith("top_") or len(frame) > 8:
        ordered = frame.sort_values(y).tail(15)
        figure.add_trace(go.Bar(x=ordered[y], y=ordered[x], orientation="h"))
    else:
        figure.add_trace(go.Bar(x=frame[x], y=frame[y]))

    figure.update_layout(title=name.replace("_", " ").title(),
                         template="plotly_white", height=400)
    return figure


def _summarize(name: str, frame: pd.DataFrame) -> str:
    """Produce a short textual summary for a gold table."""
    numeric_cols = [c for c in frame.columns
                    if pd.api.types.is_numeric_dtype(frame[c]) and not c.startswith("_")]
    parts = [f"{len(frame)} rows"]
    for col in numeric_cols[:2]:
        total = float(frame[col].sum())
        parts.append(f"total {col} = {total:,.2f}")
    return f"{name}: " + ", ".join(parts)


def run_reporter(gold_parquet_paths: list[str], business_intent: str,
                 run_id: str) -> str:
    """Render a standalone HTML report and return its path."""
    if not gold_parquet_paths:
        raise ValueError("gold_parquet_paths must not be empty")
    if not run_id:
        raise ValueError("run_id must be provided")

    audit = AuditLogger(run_id)
    trace = AgentTrace("reporter", run_id)
    trace.set_input({"gold_paths": gold_parquet_paths, "intent": business_intent})
    audit.log("reporter", "report", "start", {"tables": len(gold_parquet_paths)})

    try:
        sections: list[str] = []
        summaries: list[str] = []

        for path in gold_parquet_paths:
            if not Path(path).exists():
                continue
            name = _table_name_from_path(path, run_id)
            frame = _load_table(path)
            summary = _summarize(name, frame)
            summaries.append(summary)
            trace.add_tool_call("load_table", {"path": path}, len(frame))

            figure = _build_chart(name, frame)
            chart_html = (
                to_html(figure, include_plotlyjs=False, full_html=False)
                if figure is not None
                else "<p>No chart available for this table.</p>"
            )
            table_html = frame.head(20).to_html(index=False, border=0)
            sections.append(
                f"<section><h2>{html.escape(name.replace('_', ' ').title())}</h2>"
                f"<p>{html.escape(summary)}</p>{chart_html}"
                f"<details><summary>Data preview</summary>{table_html}</details></section>"
            )

        document = _render_document(business_intent, run_id, summaries, sections)
        out_path = Path(config.REPORTS_DIR) / f"report_{run_id}.html"
        out_path.write_text(document, encoding="utf-8")
        trace.add_verification(f"rendered {len(sections)} sections")

        try:
            VectorMemory().store_report(
                " | ".join(summaries) or f"report {run_id}",
                metadata={"run_id": run_id, "intent": business_intent},
            )
        except Exception:
            trace.add_verification("vector memory unavailable; report summary not stored")

        audit.log("reporter", "report", "complete", {"report_path": str(out_path)})
        trace.finish(output={"report_path": str(out_path)})
        return str(out_path)
    except Exception as exc:
        audit.log("reporter", "report", "error", {"error": str(exc)})
        trace.finish(output={"error": str(exc)}, status="error")
        raise


def _render_document(business_intent: str, run_id: str, summaries: list[str],
                     sections: list[str]) -> str:
    """Assemble the full HTML document string."""
    overview = "".join(f"<li>{html.escape(s)}</li>" for s in summaries)
    body = "".join(sections)
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>Retail Medallion Report {html.escape(run_id)}</title>"
        "<script src=\"https://cdn.plot.ly/plotly-2.32.0.min.js\"></script>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:2rem;color:#1a1a1a;}"
        "section{margin-bottom:2.5rem;border-bottom:1px solid #eee;padding-bottom:1rem;}"
        "table{border-collapse:collapse;font-size:0.85rem;}"
        "th,td{border:1px solid #ddd;padding:4px 8px;}"
        "h1{color:#b8860b;}h2{color:#333;}</style></head><body>"
        f"<h1>Retail Medallion Pipeline Report</h1>"
        f"<p><strong>Run:</strong> {html.escape(run_id)}</p>"
        f"<p><strong>Business intent:</strong> {html.escape(business_intent or 'n/a')}</p>"
        f"<h2>Overview</h2><ul>{overview}</ul>{body}"
        "</body></html>"
    )
