"""Tests for the gold aggregation agent."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.gold_agent import run_gold_agent
from agents.sttm_generator import run_sttm_generator


def _make_silver(path: Path) -> None:
    """Write a small silver parquet with numeric and dimension columns."""
    frame = pd.DataFrame(
        {
            "order_id": ["1", "2", "3", "4"],
            "customer_id": ["C1", "C2", "C1", "C3"],
            "product": ["alpha", "beta", "alpha", "gamma"],
            "order_date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-10"],
            "total_amount": [20.0, 40.0, 15.0, 30.0],
            "quantity": [2, 4, 1, 3],
            "_run_id": ["runG"] * 4,
        }
    )
    frame.to_parquet(path, index=False)


def test_gold_generates_tables_with_metadata(tmp_path):
    """Gold produces aggregated tables that carry the run id metadata."""
    silver_path = tmp_path / "silver.parquet"
    _make_silver(silver_path)

    gold_sttm = run_sttm_generator(
        "", "gold", "revenue analysis", "runG",
        silver_parquet_path=str(silver_path),
    )
    assert Path(gold_sttm).exists()

    outputs = run_gold_agent([str(silver_path)], gold_sttm, "runG")
    assert outputs

    for path in outputs:
        frame = pd.read_parquet(path)
        assert "_run_id" in frame.columns
        assert (frame["_run_id"] == "runG").all()

    names = [Path(p).stem for p in outputs]
    assert any("revenue_by_" in n for n in names)


def test_gold_requires_inputs():
    """Gold rejects empty silver inputs."""
    try:
        run_gold_agent([], "x.csv", "runG")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
