"""Tests for the bronze ingestion agent."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.bronze_agent import run_bronze_agent


def _write_bronze_sttm(path: Path) -> None:
    """Write a minimal bronze STTM for the sample columns."""
    rows = [
        ("order_id", "order_id", "string", "passthrough", "id"),
        ("product", "product", "string", "title_case", "name"),
        ("unit_price", "unit_price", "float", "strip_currency", "price"),
        ("quantity", "quantity", "float", "cast_numeric", "qty"),
        ("total_amount", "total_amount", "float", "strip_currency", "total"),
    ]
    pd.DataFrame(
        rows,
        columns=["source_column", "target_column", "target_type",
                 "transformation", "description"],
    ).to_csv(path, index=False)


def test_bronze_strips_currency_and_casts(sample_csv, tmp_path):
    """Bronze applies currency stripping, numeric casting and metadata."""
    sttm_path = tmp_path / "sttm_bronze.csv"
    _write_bronze_sttm(sttm_path)

    outputs = run_bronze_agent([sample_csv], str(sttm_path), "runB")
    assert len(outputs) == 1

    frame = pd.read_parquet(outputs[0])
    assert frame["unit_price"].dtype.kind == "f"
    assert frame["unit_price"].iloc[0] == 10.0
    assert frame["quantity"].iloc[1] == 4
    assert frame["product"].iloc[0] == "Alpha"
    for meta in ("_source_file", "_ingested_at", "_run_id"):
        assert meta in frame.columns
    assert (frame["_run_id"] == "runB").all()


def test_bronze_requires_paths():
    """Bronze rejects an empty CSV path list."""
    try:
        run_bronze_agent([], "x.csv", "runB")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
