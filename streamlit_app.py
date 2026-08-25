"""Streamlit UI for the agentic retail medallion pipeline."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from agents._llm import LLMError, call_llm, extract_json
from agents.orchestrator import PipelineOrchestrator
from core import config

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def sanitize_filename(name: str) -> str:
    """Return an ASCII-safe file name limited to a known character set."""
    base = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    cleaned = cleaned.strip("._") or "upload.csv"
    if not cleaned.lower().endswith(".csv"):
        cleaned = f"{cleaned}.csv"
    return cleaned


def suggest_intents(sample_df: pd.DataFrame) -> list[str]:
    """Ask the configured LLM for three concise business intents.

    Works for both OpenAI-compatible and Claude providers and parses a JSON
    array robustly even when the model wraps it in extra prose. Raises
    :class:`RuntimeError` with a clear message on failure.
    """
    columns = list(sample_df.columns)
    preview = sample_df.head(5).to_dict(orient="records")
    system = (
        "You are a retail data analyst. Given CSV columns and sample rows, "
        "propose exactly three concise business analysis intents. Respond with "
        "a JSON array of three short strings and nothing else."
    )
    user = json.dumps({"columns": columns, "sample_rows": preview})
    try:
        raw = call_llm(system, user, max_tokens=512)
    except LLMError as exc:
        raise RuntimeError(str(exc)) from exc

    try:
        parsed = extract_json(raw)
    except ValueError as exc:
        raise RuntimeError(f"could not parse model output: {raw[:200]}") from exc

    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                parsed = value
                break
    if not isinstance(parsed, list):
        raise RuntimeError("model did not return a list of intents")
    intents = [str(item).strip() for item in parsed if str(item).strip()]
    if not intents:
        raise RuntimeError("model returned no usable intents")
    return intents[:3]


def _init_state() -> None:
    """Initialize persistent Streamlit session state keys."""
    defaults = {
        "orchestrator": None,
        "run_id": "",
        "runtime_phase": "idle",
        "csv_path": "",
        "business_intent": "",
        "suggested_intents": [],
        "last_error": "",
        "report_path": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _orchestrator() -> PipelineOrchestrator:
    """Return a cached orchestrator instance."""
    if st.session_state["orchestrator"] is None:
        st.session_state["orchestrator"] = PipelineOrchestrator()
    return st.session_state["orchestrator"]


def _phase_index(runtime_phase: str) -> int:
    """Map a runtime phase to a 0-4 progress index."""
    order = {
        "idle": 0,
        "pending_run": 0,
        "hitl_bronze": 1,
        "pending_phase2": 1,
        "hitl_silver": 2,
        "pending_phase3": 2,
        "hitl_gold": 3,
        "pending_phase4": 3,
        "gold_data_ready": 4,
        "phase_4_complete": 4,
        "error": 0,
    }
    return order.get(runtime_phase, 0)


def _sync_runtime_from_state(state: dict) -> None:
    """Update the runtime phase from the orchestrator's next node."""
    pending = state.get("_next", [])
    if "hitl_bronze" in pending:
        st.session_state["runtime_phase"] = "hitl_bronze"
    elif "hitl_silver" in pending:
        st.session_state["runtime_phase"] = "hitl_silver"
    elif "hitl_gold" in pending:
        st.session_state["runtime_phase"] = "hitl_gold"
    elif state.get("gold_parquet_paths"):
        st.session_state["runtime_phase"] = "gold_data_ready"
    else:
        st.session_state["runtime_phase"] = "pending_run"


def _render_sidebar() -> None:
    """Render the sidebar controls for upload, intent and run actions."""
    with st.sidebar:
        st.header("Configuration")
        st.caption(f"Provider: {config.LLM_PROVIDER} | Model: {config.LLM_MODEL}")

        uploaded = st.file_uploader("Upload orders CSV (max 50 MB)", type=["csv"])
        if uploaded is not None:
            if uploaded.size > MAX_UPLOAD_BYTES:
                st.error("File exceeds the 50 MB limit.")
            else:
                safe_name = sanitize_filename(uploaded.name)
                dest = Path(config.UPLOADS_DIR) / safe_name
                dest.write_bytes(uploaded.getbuffer())
                st.session_state["csv_path"] = str(dest)
                st.success(f"Saved {safe_name}")

        st.session_state["business_intent"] = st.text_area(
            "Business intent", value=st.session_state["business_intent"], height=90
        )

        if st.button("Analyze File (suggest intents)"):
            _handle_analyze()

        for idx, intent in enumerate(st.session_state["suggested_intents"]):
            if st.button(f"Use: {intent}", key=f"intent_{idx}"):
                st.session_state["business_intent"] = intent

        run_disabled = not st.session_state["csv_path"]
        if st.button("Run Pipeline", disabled=run_disabled):
            _handle_run()

        if st.button("Reset"):
            _handle_reset()


def _handle_analyze() -> None:
    """Handle the analyze-file action and surface any errors clearly."""
    csv_path = st.session_state["csv_path"]
    if not csv_path:
        st.session_state["last_error"] = "Upload a CSV before analyzing."
        return
    try:
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        st.session_state["suggested_intents"] = suggest_intents(frame)
        st.session_state["last_error"] = ""
    except Exception as exc:
        st.session_state["suggested_intents"] = []
        st.session_state["last_error"] = f"Analyze failed: {exc}"


def _handle_run() -> None:
    """Start a new pipeline run."""
    try:
        run_id = uuid.uuid4().hex[:12]
        st.session_state["run_id"] = run_id
        st.session_state["runtime_phase"] = "pending_run"
        state = _orchestrator().start_pipeline(
            run_id,
            [st.session_state["csv_path"]],
            st.session_state["business_intent"],
        )
        _sync_runtime_from_state(state)
        st.session_state["last_error"] = ""
    except Exception as exc:
        st.session_state["runtime_phase"] = "error"
        st.session_state["last_error"] = f"Run failed: {exc}"


def _handle_reset() -> None:
    """Reset the session to its initial state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _init_state()


def _render_hitl(layer: str, sttm_key: str, next_phase_label: str) -> None:
    """Render a human-in-the-loop review panel for the given layer."""
    orch = _orchestrator()
    run_id = st.session_state["run_id"]
    state = orch.get_pipeline_state(run_id)
    sttm_path = state.get(sttm_key, "")

    st.subheader(f"Review {layer.title()} STTM")
    if sttm_path and Path(sttm_path).exists():
        sttm_df = pd.read_csv(sttm_path, dtype=str, keep_default_na=False)
        st.dataframe(sttm_df, use_container_width=True)
    else:
        st.info("STTM not available yet.")

    prior = state.get(f"{layer}_feedback", [])
    if prior:
        with st.expander(f"Prior rejection rounds ({len(prior)})"):
            for i, note in enumerate(prior, start=1):
                st.write(f"{i}. {note}")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(f"Approve {layer.title()}", key=f"approve_{layer}"):
            new_state = orch.approve_sttm(run_id)
            _sync_runtime_from_state(new_state)
            st.rerun()
    with col_b:
        feedback = st.text_input("Rejection feedback", key=f"feedback_{layer}")
        if st.button(f"Reject {layer.title()}", key=f"reject_{layer}"):
            if not feedback.strip():
                st.warning("Provide feedback before rejecting.")
            else:
                new_state = orch.reject_sttm(run_id, feedback.strip())
                _sync_runtime_from_state(new_state)
                st.rerun()


def _render_previews() -> None:
    """Render raw, profile, bronze, silver and gold previews."""
    orch = _orchestrator()
    run_id = st.session_state["run_id"]
    if not run_id:
        return
    state = orch.get_pipeline_state(run_id)

    with st.expander("Raw CSV preview"):
        if st.session_state["csv_path"]:
            st.dataframe(
                pd.read_csv(st.session_state["csv_path"], dtype=str,
                            keep_default_na=False).head(20),
                use_container_width=True,
            )

    profile_path = state.get("profile_path", "")
    if profile_path and Path(profile_path).exists():
        with st.expander("Profile summary"):
            profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            st.write(profile.get("quality_notes", []))
            st.json(profile.get("semantic_meanings", {}))

    for label, key in [
        ("Bronze parquet", "bronze_parquet_paths"),
        ("Silver parquet", "silver_parquet_path"),
    ]:
        value = state.get(key)
        paths = value if isinstance(value, list) else ([value] if value else [])
        for path in paths:
            if path and Path(path).exists():
                with st.expander(f"{label}: {Path(path).name}"):
                    st.dataframe(pd.read_parquet(path).head(20),
                                 use_container_width=True)

    gold_paths = state.get("gold_parquet_paths", [])
    if gold_paths:
        st.subheader("Gold outputs")
        for path in gold_paths:
            if Path(path).exists():
                with st.expander(Path(path).name):
                    st.dataframe(pd.read_parquet(path), use_container_width=True)


def _render_gold_actions() -> None:
    """Render gold revision and report generation controls."""
    orch = _orchestrator()
    run_id = st.session_state["run_id"]
    state = orch.get_pipeline_state(run_id)
    if not state.get("gold_parquet_paths"):
        return

    st.subheader("Gold revision")
    revision = st.text_input("Describe changes to the gold analysis",
                             key="gold_revision")
    if st.button("Rerun Gold aggregation"):
        if not revision.strip():
            st.warning("Describe the requested change first.")
        else:
            try:
                orch.revise_gold(run_id, revision.strip())
                st.success("Gold aggregation updated.")
                st.rerun()
            except Exception as exc:
                st.error(f"Revision failed: {exc}")

    st.subheader("Report")
    if st.button("Generate Report"):
        try:
            report_path = orch.generate_report(run_id)
            st.session_state["report_path"] = report_path
            st.success("Report generated.")
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")

    report_path = st.session_state.get("report_path", "")
    if report_path and Path(report_path).exists():
        html_text = Path(report_path).read_text(encoding="utf-8")
        st.components.v1.html(html_text, height=600, scrolling=True)
        st.download_button(
            "Download report",
            data=html_text,
            file_name=Path(report_path).name,
            mime="text/html",
        )


def main() -> None:
    """Entry point for the Streamlit application."""
    st.set_page_config(page_title="Retail Medallion Pipeline", layout="wide")
    _init_state()

    st.title("Retail Medallion Pipeline")
    st.caption("Bronze -> Silver -> Gold with human-in-the-loop STTM approvals")

    _render_sidebar()

    if st.session_state["last_error"]:
        st.error(st.session_state["last_error"])
        st.info(
            "Troubleshooting: confirm your API key/token is set, that the "
            "LLM_MODEL is accessible for your account (a 404 means switch to an "
            "available model ID), and that the uploaded CSV is well-formed."
        )

    phase = _phase_index(st.session_state["runtime_phase"])
    st.progress(phase / 4.0, text=f"Phase {phase} of 4")

    runtime = st.session_state["runtime_phase"]
    if runtime == "hitl_bronze":
        _render_hitl("bronze", "sttm_bronze_path", "Phase 2")
    elif runtime == "hitl_silver":
        _render_hitl("silver", "sttm_silver_path", "Phase 3")
    elif runtime == "hitl_gold":
        _render_hitl("gold", "sttm_gold_path", "Phase 4")

    _render_previews()

    if st.session_state["run_id"]:
        _render_gold_actions()


if __name__ == "__main__":
    main()
