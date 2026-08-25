"""Tests for core utilities: audit, tracing, memory and state."""

from __future__ import annotations

from core.audit import AuditLogger
from core.memory import VectorMemory
from core.observability import AgentTrace
from core.state import PipelineState


def test_audit_logger_appends_jsonl():
    """AuditLogger writes one JSON record per event."""
    logger = AuditLogger("run1")
    logger.log("agent", "phase_1", "start", {"a": 1})
    logger.log("agent", "phase_1", "done", {"b": 2})
    records = logger.read_all()
    assert len(records) == 2
    assert records[0]["run_id"] == "run1"
    assert records[0]["event"] == "start"
    assert records[1]["details"] == {"b": 2}


def test_audit_logger_requires_run_id():
    """AuditLogger rejects an empty run id."""
    try:
        AuditLogger("")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_agent_trace_persists_json():
    """AgentTrace finalizes and writes a JSON trace file."""
    trace = AgentTrace("profiler", "run2")
    trace.set_input({"x": 1})
    trace.add_plan("step one")
    trace.add_tool_call("tool", {"k": "v"}, "ok")
    trace.add_verification("checked")
    record = trace.finish(output={"done": True})
    assert record["status"] == "success"
    assert record["duration"] >= 0
    assert trace.path.exists()


def test_vector_memory_store_and_retrieve():
    """VectorMemory stores intents and retrieves similar entries."""
    memory = VectorMemory()
    memory.store_intent("analyze revenue by product")
    memory.store_intent("track shipping delays")
    results = memory.retrieve_similar_intents("revenue by product", k=1)
    assert isinstance(results, list)
    assert memory.backend in {"chroma", "memory"}


def test_pipeline_state_roundtrip():
    """PipelineState serializes and deserializes without loss."""
    state = PipelineState(run_id="r", business_intent="intent",
                          csv_paths=["a.csv"], bronze_feedback=["fix"])
    data = state.to_dict()
    restored = PipelineState.from_dict(data)
    assert restored.run_id == "r"
    assert restored.csv_paths == ["a.csv"]
    assert restored.bronze_feedback == ["fix"]


def test_pipeline_state_from_dict_ignores_unknown():
    """from_dict ignores unexpected keys."""
    restored = PipelineState.from_dict({"run_id": "r", "unknown": 1})
    assert restored.run_id == "r"
