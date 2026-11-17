"""Central configuration and current model defaults for the evaluation pipeline.

Model IDs are intentionally kept in one place so the agent, DeepEval judges and
Promptfoo configuration can be upgraded together. Defaults reflect the models
available in November 2026.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Latest model defaults (November 2026)
# ---------------------------------------------------------------------------
DEFAULT_AGENT_MODELS: dict[str, str] = {
    "gemini": "gemini-3.8-flash",
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-6.1-terra",
}

# Availability-first fallbacks for live tool-calling runs. Gemini's flagship
# tiers are frequently saturated during peak hours, so the lighter flash tier
# keeps demos and smoke tests working without changing the default model.
DEFAULT_AGENT_FALLBACKS: dict[str, list[str]] = {
    "gemini": ["gemini-flash-lite-latest", "gemini-3.6-flash"],
    "groq": ["openai/gpt-oss-20b"],
    "openai": ["gpt-6-luna"],
}

DEFAULT_JUDGE_MODELS: dict[str, str] = {
    "gemini": "gemini-3.8-flash",
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-6.1-terra",
}

# Tried in order when the primary judge model is unavailable or rate limited.
# Availability-first ordering: high-end Gemini tiers are frequently saturated,
# and the judge only needs reliable structured output.
DEFAULT_JUDGE_FALLBACKS: dict[str, list[str]] = {
    "gemini": ["gemini-flash-lite-latest", "gemini-3.6-flash", "gemini-4-hydrogen"],
    "groq": ["openai/gpt-oss-20b"],
    "openai": ["gpt-6-luna", "gpt-6.1-sol"],
}

OPENAI_COMPATIBLE_BASE_URLS: dict[str, Optional[str]] = {
    # Google exposes an OpenAI-compatible endpoint. It supports tool calling and
    # JSON mode for the current Gemini models.
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "openai": None,
}

API_KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
}

SUPPORTED_PROVIDERS = tuple(DEFAULT_JUDGE_MODELS)


def get_api_key(provider: str) -> Optional[str]:
    """Return the first configured API key for *provider*."""
    for env_var in API_KEY_ENV_VARS.get(provider, ()):
        value = os.getenv(env_var)
        if value and value.strip():
            return value.strip()
    return None


@dataclass(frozen=True)
class AgentSettings:
    """Runtime configuration for :class:`src.agent.core_agent.SupportAgent`."""

    mode: str = "deterministic"
    provider: str = "gemini"
    model: str = "gemini-3.8-flash"
    fallbacks: tuple[str, ...] = ()
    max_steps: int = 6
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "AgentSettings":
        mode = os.getenv("AGENT_MODE", "deterministic").strip().lower()
        if mode in {"live", "llm"}:
            mode = "llm"
        elif mode != "deterministic":
            mode = "deterministic"

        provider = os.getenv("AGENT_PROVIDER", "gemini").strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            provider = "gemini"

        model = os.getenv("AGENT_MODEL", "").strip() or DEFAULT_AGENT_MODELS[provider]

        fallbacks = tuple(
            item.strip()
            for item in os.getenv("AGENT_FALLBACKS", "").split(",")
            if item.strip()
        ) or tuple(DEFAULT_AGENT_FALLBACKS.get(provider, []))

        try:
            max_steps = int(os.getenv("AGENT_MAX_STEPS", "6"))
        except ValueError:
            max_steps = 6
        max_steps = max(1, min(max_steps, 12))

        try:
            temperature = float(os.getenv("AGENT_TEMPERATURE", "0"))
        except ValueError:
            temperature = 0.0

        return cls(
            mode=mode,
            provider=provider,
            model=model,
            fallbacks=fallbacks,
            max_steps=max_steps,
            temperature=max(0.0, min(temperature, 2.0)),
        )


@dataclass(frozen=True)
class JudgeSettings:
    """Configuration for the DeepEval judge chain."""

    provider: str = "gemini"
    model: str = "gemini-3.8-flash"
    fallbacks: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "JudgeSettings":
        provider = os.getenv("EVAL_JUDGE_PROVIDER", "gemini").strip().lower()
        if provider not in SUPPORTED_PROVIDERS and provider != "mock":
            provider = "gemini"

        model = (
            os.getenv("EVAL_JUDGE_MODEL", "").strip()
            or DEFAULT_JUDGE_MODELS.get(provider, "gemini-3.8-flash")
        )

        raw_fallbacks = os.getenv("EVAL_JUDGE_FALLBACKS", "").strip()
        fallbacks: tuple[str, ...]
        if raw_fallbacks:
            fallbacks = tuple(
                item.strip() for item in raw_fallbacks.split(",") if item.strip()
            )
        else:
            fallbacks = tuple(DEFAULT_JUDGE_FALLBACKS.get(provider, []))

        return cls(provider=provider, model=model, fallbacks=fallbacks)
