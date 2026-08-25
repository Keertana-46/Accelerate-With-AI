"""Shared LLM invocation helpers for OpenAI-compatible and Claude providers."""

from __future__ import annotations

import json
import re
from typing import Any

from core import config


class LLMError(RuntimeError):
    """Raised when an LLM call cannot be completed."""


def extract_json(text: str) -> Any:
    """Extract the first JSON object or array embedded in ``text``.

    The function tolerates code fences and surrounding prose. It raises
    :class:`ValueError` when no JSON payload can be located.
    """
    if text is None:
        raise ValueError("empty response text")
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON payload found in model response")


def call_llm(system: str, user: str, max_tokens: int = 2048) -> str:
    """Call the configured LLM and return the raw text response.

    Routes to the Anthropic SDK when the provider is Claude, otherwise uses an
    OpenAI-compatible chat completion endpoint. ``claude-sonnet-5`` is called
    without a temperature parameter as required by the provider.
    """
    if config.is_claude():
        return _call_claude(system, user, max_tokens)
    return _call_openai(system, user, max_tokens)


def _call_claude(system: str, user: str, max_tokens: int) -> str:
    """Invoke Anthropic's messages API and return the text content."""
    try:
        import anthropic  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise LLMError("anthropic package is not installed") from exc

    if not config.CLAUDE_API_KEY:
        raise LLMError("CLAUDE_API_KEY is not configured")

    client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
    kwargs: dict[str, Any] = {
        "model": config.LLM_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if not config.skip_temperature():
        kwargs["temperature"] = config.LLM_TEMPERATURE

    try:
        response = client.messages.create(**kwargs)
    except Exception as exc:
        raise LLMError(f"Claude request failed: {exc}") from exc

    parts = [block.text for block in response.content if getattr(block, "text", None)]
    return "".join(parts)


def _call_openai(system: str, user: str, max_tokens: int) -> str:
    """Invoke an OpenAI-compatible chat completion and return the text."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise LLMError("openai package is not installed") from exc

    if not config.OPENAI_API_KEY:
        raise LLMError("OPENAI_API_KEY is not configured")

    client_kwargs: dict[str, Any] = {"api_key": config.OPENAI_API_KEY}
    if config.LLM_BASE_URL:
        client_kwargs["base_url"] = config.LLM_BASE_URL
    client = OpenAI(**client_kwargs)

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as exc:
        raise LLMError(f"OpenAI-compatible request failed: {exc}") from exc

    return response.choices[0].message.content or ""
