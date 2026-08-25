"""Tests for the profiler agent tools and profile generation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents import profiler


def test_inspect_files(sample_csv):
    """inspect_files reports shape, columns and missing counts."""
    result = profiler.inspect_files(sample_csv)
    assert result["shape"][0] == 4
    assert "order_id" in result["columns"]
    assert "missing_counts" in result


def test_profiler_detects_currency(sample_csv):
    """profiler_tool flags currency formatted columns."""
    stats = profiler.profiler_tool(sample_csv)
    flags = stats["columns"]["unit_price"]["quality_flags"]
    assert "currency_formatted" in flags


def test_profiler_detects_mixed_dates(sample_csv):
    """profiler_tool flags mixed date formats."""
    stats = profiler.profiler_tool(sample_csv)
    flags = stats["columns"]["order_date"]["quality_flags"]
    assert "mixed_date_formats" in flags


def test_run_profiler_writes_profile(sample_csv, monkeypatch):
    """run_profiler writes a profile JSON without requiring an LLM."""
    monkeypatch.setattr(profiler, "_semantic_meanings", lambda cols, intent: {})
    profile_path = profiler.run_profiler(sample_csv, "analyze revenue", "runP")
    assert Path(profile_path).exists()
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    assert profile["run_id"] == "runP"
    assert profile["quality_notes"]


@pytest.mark.skipif(
    not (os.getenv("RUN_LLM_TESTS") and (os.getenv("CLAUDE_API_KEY")
         or os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN"))),
    reason="LLM credentials not configured",
)
def test_semantic_meanings_live(sample_csv):
    """Semantic meaning generation returns a mapping when an LLM is available."""
    meanings = profiler._semantic_meanings(["order_id", "product"], "sales")
    assert isinstance(meanings, dict)
