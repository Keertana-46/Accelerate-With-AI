"""LangGraph orchestration for the medallion pipeline with HITL gates."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.bronze_agent import run_bronze_agent
from agents.gold_agent import run_gold_agent
from agents.profiler import run_profiler
from agents.reporter import run_reporter
from agents.silver_agent import run_silver_agent
from agents.sttm_generator import run_sttm_generator
from core.audit import AuditLogger


class GraphState(TypedDict, total=False):
    """LangGraph channel schema mirroring :class:`PipelineState`."""

    run_id: str
    business_intent: str
    csv_paths: list[str]
    profile_path: str
    sttm_bronze_path: str
    sttm_silver_path: str
    sttm_gold_path: str
    bronze_parquet_paths: list[str]
    silver_parquet_path: str
    gold_parquet_paths: list[str]
    report_path: str
    bronze_approved: bool
    silver_approved: bool
    gold_approved: bool
    bronze_feedback: list[str]
    silver_feedback: list[str]
    gold_feedback: list[str]
    current_phase: str
    errors: list[str]


def _phase_1(state: GraphState) -> dict[str, Any]:
    """Profile the source data and generate the Bronze STTM."""
    run_id = state["run_id"]
    intent = state.get("business_intent", "")
    csv_paths = state.get("csv_paths", [])
    feedback = "; ".join(state.get("bronze_feedback", []))

    profile_path = run_profiler(csv_paths[0], intent, run_id)
    sttm_path = run_sttm_generator(
        profile_path, "bronze", intent, run_id, reviewer_feedback=feedback
    )
    return {
        "profile_path": profile_path,
        "sttm_bronze_path": sttm_path,
        "current_phase": "hitl_bronze",
    }


def _hitl_bronze(state: GraphState) -> dict[str, Any]:
    """Human-in-the-loop gate for the Bronze STTM."""
    decision = interrupt(
        {"layer": "bronze", "sttm_path": state.get("sttm_bronze_path", "")}
    )
    if decision.get("approved"):
        return {"bronze_approved": True}
    feedback = list(state.get("bronze_feedback", []))
    if decision.get("feedback"):
        feedback.append(decision["feedback"])
    return {"bronze_approved": False, "bronze_feedback": feedback}


def _phase_2(state: GraphState) -> dict[str, Any]:
    """Run the Bronze agent and generate the Silver STTM."""
    run_id = state["run_id"]
    intent = state.get("business_intent", "")
    bronze_paths = run_bronze_agent(
        state.get("csv_paths", []), state["sttm_bronze_path"], run_id
    )
    sttm_silver = run_sttm_generator(
        state.get("profile_path", ""),
        "silver",
        intent,
        run_id,
        prev_sttm_path=state["sttm_bronze_path"],
        reviewer_feedback="; ".join(state.get("silver_feedback", [])),
    )
    return {
        "bronze_parquet_paths": bronze_paths,
        "sttm_silver_path": sttm_silver,
        "current_phase": "hitl_silver",
    }


def _hitl_silver(state: GraphState) -> dict[str, Any]:
    """Human-in-the-loop gate for the Silver STTM."""
    decision = interrupt(
        {"layer": "silver", "sttm_path": state.get("sttm_silver_path", "")}
    )
    if decision.get("approved"):
        return {"silver_approved": True}
    feedback = list(state.get("silver_feedback", []))
    if decision.get("feedback"):
        feedback.append(decision["feedback"])
    return {"silver_approved": False, "silver_feedback": feedback}


def _phase_3(state: GraphState) -> dict[str, Any]:
    """Run the Silver agent and generate the Gold STTM."""
    run_id = state["run_id"]
    intent = state.get("business_intent", "")
    silver_path = run_silver_agent(
        state.get("bronze_parquet_paths", []), state["sttm_silver_path"], run_id
    )
    sttm_gold = run_sttm_generator(
        state.get("profile_path", ""),
        "gold",
        intent,
        run_id,
        silver_parquet_path=silver_path,
        reviewer_feedback="; ".join(state.get("gold_feedback", [])),
    )
    return {
        "silver_parquet_path": silver_path,
        "sttm_gold_path": sttm_gold,
        "current_phase": "hitl_gold",
    }


def _hitl_gold(state: GraphState) -> dict[str, Any]:
    """Human-in-the-loop gate for the Gold STTM."""
    decision = interrupt(
        {"layer": "gold", "sttm_path": state.get("sttm_gold_path", "")}
    )
    if decision.get("approved"):
        return {"gold_approved": True}
    feedback = list(state.get("gold_feedback", []))
    if decision.get("feedback"):
        feedback.append(decision["feedback"])
    return {"gold_approved": False, "gold_feedback": feedback}


def _phase_4(state: GraphState) -> dict[str, Any]:
    """Run the Gold agent to produce analytics-ready tables."""
    run_id = state["run_id"]
    gold_paths = run_gold_agent(
        [state["silver_parquet_path"]], state["sttm_gold_path"], run_id
    )
    return {"gold_parquet_paths": gold_paths, "current_phase": "phase_4_complete"}


def _route_bronze(state: GraphState) -> str:
    """Route out of the Bronze gate based on the approval flag."""
    return "phase_2" if state.get("bronze_approved") else "phase_1"


def _route_silver(state: GraphState) -> str:
    """Route out of the Silver gate based on the approval flag."""
    return "phase_3" if state.get("silver_approved") else "phase_2"


def _route_gold(state: GraphState) -> str:
    """Route out of the Gold gate based on the approval flag."""
    return "phase_4" if state.get("gold_approved") else "phase_3"


def _build_graph() -> Any:
    """Construct and compile the medallion StateGraph."""
    graph = StateGraph(GraphState)
    graph.add_node("phase_1", _phase_1)
    graph.add_node("hitl_bronze", _hitl_bronze)
    graph.add_node("phase_2", _phase_2)
    graph.add_node("hitl_silver", _hitl_silver)
    graph.add_node("phase_3", _phase_3)
    graph.add_node("hitl_gold", _hitl_gold)
    graph.add_node("phase_4", _phase_4)

    graph.add_edge(START, "phase_1")
    graph.add_edge("phase_1", "hitl_bronze")
    graph.add_conditional_edges(
        "hitl_bronze", _route_bronze, {"phase_2": "phase_2", "phase_1": "phase_1"}
    )
    graph.add_edge("phase_2", "hitl_silver")
    graph.add_conditional_edges(
        "hitl_silver", _route_silver, {"phase_3": "phase_3", "phase_2": "phase_2"}
    )
    graph.add_edge("phase_3", "hitl_gold")
    graph.add_conditional_edges(
        "hitl_gold", _route_gold, {"phase_4": "phase_4", "phase_3": "phase_3"}
    )
    graph.add_edge("phase_4", END)

    return graph.compile(checkpointer=MemorySaver())


class PipelineOrchestrator:
    """Drive the medallion pipeline and mediate human approvals.

    Each ``run_id`` maps to an independent LangGraph thread so multiple runs
    can be tracked concurrently.
    """

    def __init__(self) -> None:
        """Compile the pipeline graph."""
        self._graph = _build_graph()

    def _config(self, run_id: str) -> dict[str, Any]:
        """Return the LangGraph thread configuration for ``run_id``."""
        return {"configurable": {"thread_id": run_id}}

    def start_pipeline(self, run_id: str, csv_paths: list[str],
                       business_intent: str) -> dict[str, Any]:
        """Start a run and execute until the first HITL gate."""
        if not run_id:
            raise ValueError("run_id must be provided")
        if not csv_paths:
            raise ValueError("csv_paths must not be empty")
        AuditLogger(run_id).log("orchestrator", "start", "start_pipeline",
                                {"files": len(csv_paths)})
        inputs: GraphState = {
            "run_id": run_id,
            "business_intent": business_intent,
            "csv_paths": list(csv_paths),
            "bronze_feedback": [],
            "silver_feedback": [],
            "gold_feedback": [],
            "current_phase": "phase_1",
        }
        self._graph.invoke(inputs, self._config(run_id))
        return self.get_pipeline_state(run_id)

    def get_pipeline_state(self, run_id: str) -> dict[str, Any]:
        """Return the current persisted state for ``run_id``."""
        snapshot = self._graph.get_state(self._config(run_id))
        state = dict(snapshot.values) if snapshot.values else {}
        state["_next"] = list(snapshot.next) if snapshot.next else []
        return state

    def get_sttm_for_review(self, run_id: str) -> dict[str, Any]:
        """Return the STTM path awaiting review at the current gate."""
        state = self.get_pipeline_state(run_id)
        pending = state.get("_next", [])
        mapping = {
            "hitl_bronze": ("bronze", "sttm_bronze_path"),
            "hitl_silver": ("silver", "sttm_silver_path"),
            "hitl_gold": ("gold", "sttm_gold_path"),
        }
        for node in pending:
            if node in mapping:
                layer, key = mapping[node]
                return {"layer": layer, "sttm_path": state.get(key, ""),
                        "run_id": run_id}
        return {"layer": "", "sttm_path": "", "run_id": run_id}

    def approve_sttm(self, run_id: str) -> dict[str, Any]:
        """Approve the pending STTM and advance to the next phase."""
        AuditLogger(run_id).log("orchestrator", "hitl", "approve", {})
        self._graph.invoke(Command(resume={"approved": True}),
                           self._config(run_id))
        return self.get_pipeline_state(run_id)

    def reject_sttm(self, run_id: str, feedback: str) -> dict[str, Any]:
        """Reject the pending STTM with feedback and loop the current phase."""
        if not feedback:
            raise ValueError("feedback must be provided when rejecting")
        AuditLogger(run_id).log("orchestrator", "hitl", "reject",
                                {"feedback": feedback})
        self._graph.invoke(
            Command(resume={"approved": False, "feedback": feedback}),
            self._config(run_id),
        )
        return self.get_pipeline_state(run_id)

    def generate_report(self, run_id: str) -> str:
        """Generate the HTML report for a completed run."""
        state = self.get_pipeline_state(run_id)
        gold_paths = state.get("gold_parquet_paths", [])
        if not gold_paths:
            raise ValueError("gold tables are not ready; complete phase 4 first")
        report_path = run_reporter(
            gold_paths, state.get("business_intent", ""), run_id
        )
        self._graph.update_state(self._config(run_id), {"report_path": report_path})
        return report_path

    def revise_gold(self, run_id: str, feedback: str) -> dict[str, Any]:
        """Re-run Gold aggregation after a revision request."""
        if not feedback:
            raise ValueError("feedback must be provided for a gold revision")
        state = self.get_pipeline_state(run_id)
        silver_path = state.get("silver_parquet_path", "")
        if not silver_path:
            raise ValueError("silver output missing; cannot revise gold")
        intent = state.get("business_intent", "")
        gold_feedback = list(state.get("gold_feedback", [])) + [feedback]
        AuditLogger(run_id).log("orchestrator", "phase_4", "revise_gold",
                                {"feedback": feedback})
        sttm_gold = run_sttm_generator(
            state.get("profile_path", ""),
            "gold",
            intent,
            run_id,
            silver_parquet_path=silver_path,
            reviewer_feedback="; ".join(gold_feedback),
        )
        gold_paths = run_gold_agent([silver_path], sttm_gold, run_id)
        self._graph.update_state(
            self._config(run_id),
            {
                "sttm_gold_path": sttm_gold,
                "gold_parquet_paths": gold_paths,
                "gold_feedback": gold_feedback,
            },
        )
        return self.get_pipeline_state(run_id)
