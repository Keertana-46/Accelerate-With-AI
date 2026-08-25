"""Tests for the silver refinement agent."""

from __future__ import annotations

import pandas as pd

from agents.bronze_agent import run_bronze_agent
from agents.silver_agent import run_silver_agent


def _write_bronze_sttm(path) -> None:
    """Write a bronze STTM covering the sample columns."""
    rows = [
        ("order_id", "order_id", "string", "passthrough", "id"),
        ("customer_id", "customer_id", "string", "passthrough", "cust"),
        ("order_date", "order_date", "string", "to_date", "ordered"),
        ("ship_date", "ship_date", "string", "to_date", "shipped"),
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


def test_silver_derives_columns_and_dedupes(sample_csv, tmp_path):
    """Silver derives fields, normalizes dates and deduplicates order ids."""
    bronze_sttm = tmp_path / "sttm_bronze.csv"
    _write_bronze_sttm(bronze_sttm)
    bronze_paths = run_bronze_agent([sample_csv], str(bronze_sttm), "runS")

    silver_sttm = tmp_path / "sttm_silver.csv"
    silver_sttm.write_text("source_column,target_column\n", encoding="utf-8")

    silver_path = run_silver_agent(bronze_paths, str(silver_sttm), "runS")
    frame = pd.read_parquet(silver_path)

    for col in ("total_amount_calculated", "amount_variance_flag", "days_to_ship"):
        assert col in frame.columns
    # order_id 3 was duplicated in the sample; dedup keeps three unique rows.
    assert frame["order_id"].nunique() == 3
    assert len(frame) == 3
    assert frame["total_amount_calculated"].iloc[0] == 20.0
    assert frame["days_to_ship"].iloc[0] == 2
    assert (frame["_run_id"] == "runS").all()
