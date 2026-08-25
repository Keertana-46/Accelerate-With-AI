"""Tests for configuration bootstrap and model configuration."""

from __future__ import annotations

from core import config


def test_directories_exist():
    """All configured data directories are created."""
    for path in (
        config.DATA_DIR,
        config.BRONZE_DIR,
        config.SILVER_DIR,
        config.GOLD_DIR,
        config.PROFILES_DIR,
        config.REPORTS_DIR,
        config.TRACES_DIR,
        config.STTM_BRONZE,
        config.STTM_SILVER,
        config.STTM_GOLD,
        config.UPLOADS_DIR,
        config.CHROMA_DIR,
    ):
        assert path.exists()


def test_model_config_keys():
    """model_config exposes the expected keys."""
    cfg = config.model_config()
    for key in ("provider", "model", "base_url", "temperature"):
        assert key in cfg


def test_provider_is_known():
    """The resolved provider is one of the supported values."""
    assert config.LLM_PROVIDER in {"claude", "openai_compatible", "openai"}


def test_skip_temperature_flag():
    """skip_temperature returns a boolean without error."""
    assert isinstance(config.skip_temperature(), bool)
