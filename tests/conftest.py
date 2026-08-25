"""Shared pytest fixtures that isolate the filesystem for hermetic tests."""

from __future__ import annotations

import pandas as pd
import pytest

from core import audit, config, observability

_DIR_ATTRS = [
    "DATA_DIR",
    "BRONZE_DIR",
    "SILVER_DIR",
    "GOLD_DIR",
    "PROFILES_DIR",
    "REPORTS_DIR",
    "TRACES_DIR",
    "STTM_DIR",
    "STTM_BRONZE",
    "STTM_SILVER",
    "STTM_GOLD",
    "UPLOADS_DIR",
    "CHROMA_DIR",
]


@pytest.fixture(autouse=True)
def isolate_dirs(tmp_path, monkeypatch):
    """Redirect all pipeline data directories under a temporary path."""
    data = tmp_path / "data"
    layout = {
        "DATA_DIR": data,
        "BRONZE_DIR": data / "bronze",
        "SILVER_DIR": data / "silver",
        "GOLD_DIR": data / "gold",
        "PROFILES_DIR": data / "profiles",
        "REPORTS_DIR": data / "reports",
        "TRACES_DIR": data / "traces",
        "STTM_DIR": data / "sttm",
        "STTM_BRONZE": data / "sttm" / "bronze",
        "STTM_SILVER": data / "sttm" / "silver",
        "STTM_GOLD": data / "sttm" / "gold",
        "UPLOADS_DIR": data / "uploads",
        "CHROMA_DIR": data / "chroma",
    }
    for name in _DIR_ATTRS:
        path = layout[name]
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, name, path, raising=False)

    monkeypatch.setattr(audit, "TRACES_DIR", layout["TRACES_DIR"], raising=False)
    monkeypatch.setattr(
        observability, "TRACES_DIR", layout["TRACES_DIR"], raising=False
    )
    yield layout


@pytest.fixture
def sample_csv(tmp_path):
    """Write a small orders CSV and return its path."""
    frame = pd.DataFrame(
        {
            "order_id": ["1", "2", "3", "3"],
            "customer_id": ["C1", "C2", "C1", "C1"],
            "order_date": ["2024-01-01", "01/02/2024", "2024-01-03", "2024-01-03"],
            "ship_date": ["2024-01-03", "01/05/2024", "2024-01-04", "2024-01-04"],
            "product": ["alpha", "beta", "gamma", "gamma"],
            "unit_price": ["$10.00", "$5.00", "$7.50", "$7.50"],
            "quantity": ["2", "4", "1", "1"],
            "total_amount": ["$20.00", "$20.00", "$7.50", "$7.50"],
        }
    )
    path = tmp_path / "orders.csv"
    frame.to_csv(path, index=False)
    return str(path)
