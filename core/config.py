"""Central configuration for the retail medallion pipeline.

Loads environment variables, resolves the active LLM provider, and creates all
data directories at import time so downstream modules can rely on their
existence.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _as_float(value: str, default: float) -> float:
    """Parse ``value`` as a float, returning ``default`` on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
PROFILES_DIR = DATA_DIR / "profiles"
REPORTS_DIR = DATA_DIR / "reports"
TRACES_DIR = DATA_DIR / "traces"
STTM_DIR = DATA_DIR / "sttm"
STTM_BRONZE = STTM_DIR / "bronze"
STTM_SILVER = STTM_DIR / "silver"
STTM_GOLD = STTM_DIR / "gold"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma")))

_ALL_DIRS = [
    DATA_DIR,
    BRONZE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    PROFILES_DIR,
    REPORTS_DIR,
    TRACES_DIR,
    STTM_DIR,
    STTM_BRONZE,
    STTM_SILVER,
    STTM_GOLD,
    UPLOADS_DIR,
    CHROMA_DIR,
]


def ensure_directories() -> None:
    """Create every configured data directory if it does not exist."""
    for directory in _ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()


# ---------------------------------------------------------------------------
# LLM provider resolution
# ---------------------------------------------------------------------------
def _resolve_provider() -> dict:
    """Resolve the active LLM provider based on available credentials.

    Priority: Claude/Anthropic -> GitHub-hosted OpenAI-compatible -> OpenAI.
    """
    claude_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or ""
    github_token = os.getenv("GITHUB_TOKEN") or ""
    openai_key = os.getenv("OPENAI_API_KEY") or ""

    env_model = os.getenv("LLM_MODEL") or ""
    env_base_url = os.getenv("LLM_BASE_URL")

    if claude_key:
        return {
            "provider": "claude",
            "model": env_model or "claude-sonnet-5",
            "base_url": env_base_url or "",
            "openai_api_key": openai_key,
            "claude_api_key": claude_key,
        }
    if github_token:
        return {
            "provider": "openai_compatible",
            "model": env_model or "gpt-4o",
            "base_url": env_base_url or "https://models.inference.ai.azure.com",
            "openai_api_key": github_token,
            "claude_api_key": "",
        }
    return {
        "provider": "openai",
        "model": env_model or "gpt-4o",
        "base_url": env_base_url or "",
        "openai_api_key": openai_key,
        "claude_api_key": "",
    }


_PROVIDER = _resolve_provider()

LLM_PROVIDER = _PROVIDER["provider"]
LLM_MODEL = _PROVIDER["model"]
LLM_BASE_URL = _PROVIDER["base_url"]
LLM_TEMPERATURE = _as_float(os.getenv("LLM_TEMPERATURE", "0"), 0.0)
OPENAI_API_KEY = _PROVIDER["openai_api_key"]
CLAUDE_API_KEY = _PROVIDER["claude_api_key"]
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def model_config() -> dict:
    """Return a snapshot of the resolved model configuration."""
    return {
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "base_url": LLM_BASE_URL,
        "temperature": LLM_TEMPERATURE,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_claude_key": bool(CLAUDE_API_KEY),
    }


def is_claude() -> bool:
    """Return ``True`` when the active provider is Anthropic Claude."""
    return LLM_PROVIDER == "claude"


def skip_temperature() -> bool:
    """Return ``True`` when the active model should not receive temperature.

    ``claude-sonnet-5`` rejects an explicit temperature parameter.
    """
    return is_claude() and "claude-sonnet-5" in (LLM_MODEL or "")
