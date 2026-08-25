"""Tests for the reporter agent and the intent suggestion parser."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.reporter import run_reporter


def _make_gold(path: Path, run_id: str) -> None:
    """Write a gold revenue table parquet file."""
    frame = pd.DataFrame(
        {
            "product": ["alpha", "beta", "gamma"],
            "total_revenue": [35.0, 40.0, 30.0],
            "_run_id": [run_id] * 3,
        }
    )
    frame.to_parquet(path, index=False)


def test_reporter_generates_html(tmp_path):
    """Reporter renders a standalone HTML file with expected content."""
    run_id = "runR"
    gold_path = tmp_path / f"gold_revenue_by_product_{run_id}.parquet"
    _make_gold(gold_path, run_id)

    report_path = run_reporter([str(gold_path)], "revenue analysis", run_id)
    assert Path(report_path).exists()

    html = Path(report_path).read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "Retail Medallion Pipeline Report" in html
    assert "Revenue By Product" in html


def test_reporter_requires_paths():
    """Reporter rejects an empty gold path list."""
    try:
        run_reporter([], "intent", "runR")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_intent_parser_handles_wrapped_json():
    """The suggestion parser extracts a JSON array from noisy model output."""
    from agents._llm import extract_json

    raw = "Here are ideas:\n```json\n[\"a\", \"b\", \"c\"]\n```\nThanks!"
    parsed = extract_json(raw)
    assert parsed == ["a", "b", "c"]


def test_intent_parser_plain_array():
    """The suggestion parser handles a bare JSON array."""
    from agents._llm import extract_json

    assert extract_json('["x", "y"]') == ["x", "y"]
