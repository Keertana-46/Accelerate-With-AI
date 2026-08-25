"""Integration tests for the LangGraph orchestrator and HITL flow.

These tests run fully offline by forcing the deterministic fallbacks: the LLM
calls used by the profiler and the bronze STTM generator are patched to raise so
the agents use their non-LLM code paths.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agents import profiler, sttm_generator
from agents._llm import LLMError
from agents.orchestrator import PipelineOrchestrator


@pytest.fixture(autouse=True)
def force_offline(monkeypatch):
    """Force deterministic, network-free behavior for all agents."""
    def _raise(*_args, **_kwargs):
        raise LLMError("offline test")

    monkeypatch.setattr(profiler, "call_llm", _raise)
    monkeypatch.setattr(sttm_generator, "call_llm", _raise)


def _orders_csv(tmp_path: Path) -> str:
    """Write a small orders CSV and return its path."""
    frame = pd.DataFrame(
        {
            "order_id": ["1", "2", "3"],
            "customer_id": ["C1", "C2", "C1"],
            "order_date": ["2024-01-01", "01/02/2024", "2024-01-03"],
            "ship_date": ["2024-01-03", "01/05/2024", "2024-01-04"],
            "product": ["alpha", "beta", "gamma"],
            "unit_price": ["$10.00", "$5.00", "$7.50"],
            "quantity": ["2", "4", "1"],
            "total_amount": ["$20.00", "$20.00", "$7.50"],
        }
    )
    path = tmp_path / "orders.csv"
    frame.to_csv(path, index=False)
    return str(path)


def test_full_pipeline_happy_path(tmp_path):
    """Approving every gate produces gold tables and an HTML report."""
    csv_path = _orders_csv(tmp_path)
    orch = PipelineOrchestrator()

    state = orch.start_pipeline("itest1", [csv_path], "revenue by product")
    assert "hitl_bronze" in state.get("_next", [])
    review = orch.get_sttm_for_review("itest1")
    assert review["layer"] == "bronze"
    assert Path(review["sttm_path"]).exists()

    state = orch.approve_sttm("itest1")
    assert "hitl_silver" in state.get("_next", [])

    state = orch.approve_sttm("itest1")
    assert "hitl_gold" in state.get("_next", [])

    state = orch.approve_sttm("itest1")
    assert state.get("gold_parquet_paths")
    assert state.get("current_phase") == "phase_4_complete"

    report_path = orch.generate_report("itest1")
    assert Path(report_path).exists()
    assert "<html" in Path(report_path).read_text(encoding="utf-8").lower()


def test_reject_then_approve_loops_bronze(tmp_path):
    """Rejecting the bronze gate loops back and records cumulative feedback."""
    csv_path = _orders_csv(tmp_path)
    orch = PipelineOrchestrator()

    orch.start_pipeline("itest2", [csv_path], "analyze shipping")
    state = orch.reject_sttm("itest2", "use clearer target names")
    # After rejection the graph regenerates and pauses again at the bronze gate.
    assert "hitl_bronze" in state.get("_next", [])
    assert "use clearer target names" in state.get("bronze_feedback", [])

    state = orch.approve_sttm("itest2")
    assert "hitl_silver" in state.get("_next", [])


def test_revise_gold_reruns_aggregation(tmp_path):
    """Revising gold reruns aggregation and records the feedback."""
    csv_path = _orders_csv(tmp_path)
    orch = PipelineOrchestrator()

    orch.start_pipeline("itest3", [csv_path], "revenue analysis")
    orch.approve_sttm("itest3")
    orch.approve_sttm("itest3")
    orch.approve_sttm("itest3")

    state = orch.revise_gold("itest3", "focus on top customers")
    assert "focus on top customers" in state.get("gold_feedback", [])
    assert state.get("gold_parquet_paths")
