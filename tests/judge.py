"""LLM-as-a-judge adapters for DeepEval.

All providers are accessed through their OpenAI-compatible chat completions
endpoints. This keeps the judge implementation small, supports the current
Gemini / Groq / OpenAI model line-ups, and lets us fail over between models and
providers when a free-tier quota is exhausted.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional, Type, Union

from pydantic import BaseModel

from deepeval.models.base_model import DeepEvalBaseLLM

from src.agent.settings import (
    DEFAULT_JUDGE_MODELS,
    OPENAI_COMPATIBLE_BASE_URLS,
    SUPPORTED_PROVIDERS,
    JudgeSettings,
    get_api_key,
)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_LOGGER = logging.getLogger(__name__)

_effective_judge_model: Optional[str] = None


def effective_judge_model() -> Optional[str]:
    """Return the judge model that most recently answered successfully."""
    return _effective_judge_model


def _mark_judge_success(judge: DeepEvalBaseLLM) -> None:
    global _effective_judge_model
    _effective_judge_model = judge.get_model_name()


def _judge_timeout() -> float:
    """Per-request timeout. Upstream 503s can otherwise hold a run for minutes."""
    try:
        return max(5.0, float(os.getenv("EVAL_JUDGE_TIMEOUT", "30")))
    except ValueError:
        return 30.0


def _judge_cooldown() -> float:
    try:
        return max(0.0, float(os.getenv("EVAL_JUDGE_COOLDOWN_SECONDS", "30")))
    except ValueError:
        return 30.0


def _judge_retry_delay() -> float:
    try:
        return max(0.5, float(os.getenv("EVAL_JUDGE_RETRY_DELAY_SECONDS", "6")))
    except ValueError:
        return 6.0


def _judge_min_interval() -> float:
    """Minimum spacing between judge requests (free tiers are RPM-limited)."""
    try:
        return max(0.0, float(os.getenv("EVAL_JUDGE_MIN_INTERVAL_SECONDS", "2")))
    except ValueError:
        return 2.0


def _judge_retry_budget() -> float:
    try:
        return max(5.0, float(os.getenv("EVAL_JUDGE_RETRY_BUDGET_SECONDS", "120")))
    except ValueError:
        return 120.0


class JudgeUnavailableError(RuntimeError):
    """Raised when no configured judge has credentials."""


def _extract_json(text: str) -> Any:
    cleaned = _JSON_FENCE_RE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Judge did not return valid JSON: {text[:300]!r}")


class OpenAICompatibleJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.provider = provider
        self.model_name = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self._sync_client = None
        self._async_client = None
        super().__init__(model)

    # DeepEvalBaseLLM ---------------------------------------------------
    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return f"{self.provider}:{self.model_name}"

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        return self._generate_sync(prompt, schema)

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        return await self._generate_async(prompt, schema)

    # Internals ---------------------------------------------------------
    def _sync_client_or_create(self):
        if self._sync_client is None:
            from openai import OpenAI

            self._sync_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=0,
                timeout=_judge_timeout(),
            )
        return self._sync_client

    def _async_client_or_create(self):
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=0,
                timeout=_judge_timeout(),
            )
        return self._async_client

    def _request_kwargs(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]],
        json_mode: bool,
    ) -> dict[str, Any]:
        full_prompt = prompt
        if schema is not None:
            full_prompt = (
                f"{prompt}\n\nReturn only a valid JSON object that conforms to this "
                f"JSON schema:\n{json.dumps(schema.model_json_schema())}"
            )

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": self.temperature,
        }
        if schema is not None and json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _parse(self, content: str, schema: Optional[Type[BaseModel]]):
        if content is None:
            raise ValueError("Judge returned an empty response.")
        if schema is None:
            return content
        return schema.model_validate(_extract_json(content))

    def _generate_sync(self, prompt: str, schema: Optional[Type[BaseModel]]):
        client = self._sync_client_or_create()

        try:
            completion = client.chat.completions.create(
                **self._request_kwargs(prompt, schema, json_mode=True)
            )
        except Exception as exc:
            # Some OpenAI-compatible providers reject response_format. Retry
            # once without JSON mode; every other failure moves on to the next
            # judge immediately instead of burning another 30s timeout.
            if "response_format" not in str(exc).lower():
                raise RuntimeError(
                    f"Judge {self.get_model_name()} failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            completion = client.chat.completions.create(
                **self._request_kwargs(prompt, schema, json_mode=False)
            )
        return self._parse(completion.choices[0].message.content or "", schema)

    async def _generate_async(self, prompt: str, schema: Optional[Type[BaseModel]]):
        client = self._async_client_or_create()

        try:
            completion = await client.chat.completions.create(
                **self._request_kwargs(prompt, schema, json_mode=True)
            )
        except Exception as exc:
            if "response_format" not in str(exc).lower():
                raise RuntimeError(
                    f"Judge {self.get_model_name()} failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            completion = await client.chat.completions.create(
                **self._request_kwargs(prompt, schema, json_mode=False)
            )
        return self._parse(completion.choices[0].message.content or "", schema)


class FallbackJudge(DeepEvalBaseLLM):
    """Try each judge in order until one returns a valid response.

    A model that fails (quota, 503, timeout, bad JSON) is parked for a short
    cooldown so every subsequent metric does not pay the same latency penalty.
    """

    def __init__(self, judges: list[DeepEvalBaseLLM]):
        if not judges:
            raise JudgeUnavailableError("No judge models were configured.")
        self.judges = judges
        self._cooldown_until: dict[int, float] = {}
        self._last_errors: dict[int, str] = {}
        self._last_success_index: Optional[int] = None
        self._last_request_at = 0.0
        super().__init__("fallback-judge")

    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return "FallbackJudge(" + " -> ".join(
            judge.get_model_name() for judge in self.judges
        ) + ")"

    def _ordered_judges(self) -> list[tuple[int, DeepEvalBaseLLM]]:
        now = time.monotonic()
        available = [
            (index, judge)
            for index, judge in enumerate(self.judges)
            if self._cooldown_until.get(index, 0.0) <= now
        ]
        if not available:
            # Every model is cooling down: the retry loop below waits for the
            # earliest cooldown instead of hammering all providers at once.
            return []

        # Once a fallback has proven healthy, keep using it instead of paying
        # the primary model's timeout on every metric.
        if self._last_success_index is not None:
            available.sort(key=lambda item: item[0] != self._last_success_index)
        return available

    def _record_failure(self, index: int, judge: DeepEvalBaseLLM, exc: Exception) -> None:
        self._last_errors[index] = f"{type(exc).__name__}: {exc}"
        self._cooldown_until[index] = time.monotonic() + _judge_cooldown()
        _LOGGER.warning(
            "Judge %s unavailable for %ss: %s",
            judge.get_model_name(),
            _judge_cooldown(),
            self._last_errors[index],
        )

    def _wait_seconds(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 0.0
        delay = _judge_retry_delay()
        if self._cooldown_until:
            earliest = min(self._cooldown_until.values())
            delay = min(delay, max(0.5, earliest - time.monotonic()))
        return max(0.5, min(delay, remaining))

    def _throttle(self) -> None:
        interval = _judge_min_interval()
        if interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request_at = time.monotonic()

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        deadline = time.monotonic() + _judge_retry_budget()
        attempt_errors: list[str] = []
        while True:
            for index, judge in self._ordered_judges():
                self._throttle()
                try:
                    result = judge.generate(prompt, schema=schema)
                    self._last_success_index = index
                    _mark_judge_success(judge)
                    return result
                except Exception as exc:
                    attempt_errors.append(
                        f"{judge.get_model_name()}: {type(exc).__name__}: {exc}"
                    )
                    self._record_failure(index, judge, exc)
            delay = self._wait_seconds(deadline)
            if delay <= 0:
                break
            time.sleep(delay)
        raise RuntimeError(
            "All judge models failed:\n"
            + ("\n".join(attempt_errors[-10:]) or "no judge was available")
        )

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        import asyncio

        deadline = time.monotonic() + _judge_retry_budget()
        attempt_errors: list[str] = []
        while True:
            for index, judge in self._ordered_judges():
                interval = _judge_min_interval()
                elapsed = time.monotonic() - self._last_request_at
                if interval > 0 and elapsed < interval:
                    await asyncio.sleep(interval - elapsed)
                self._last_request_at = time.monotonic()
                try:
                    result = await judge.a_generate(prompt, schema=schema)
                    self._last_success_index = index
                    _mark_judge_success(judge)
                    return result
                except Exception as exc:
                    attempt_errors.append(
                        f"{judge.get_model_name()}: {type(exc).__name__}: {exc}"
                    )
                    self._record_failure(index, judge, exc)
            delay = self._wait_seconds(deadline)
            if delay <= 0:
                break
            await asyncio.sleep(delay)
        raise RuntimeError(
            "All judge models failed:\n"
            + ("\n".join(attempt_errors[-10:]) or "no judge was available")
        )


def _judge_specs(settings: JudgeSettings) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = [(settings.provider, settings.model)]

    for fallback in settings.fallbacks:
        if ":" in fallback:
            prefix, candidate = fallback.split(":", 1)
            if prefix in SUPPORTED_PROVIDERS:
                specs.append((prefix, candidate))
                continue
        specs.append((settings.provider, fallback))

    # If another provider has a key configured, keep it as a last-resort
    # fallback. This makes local runs resilient to a single free-tier quota.
    if os.getenv("EVAL_JUDGE_CROSS_PROVIDER", "1").lower() not in {"0", "false", "no"}:
        for provider in SUPPORTED_PROVIDERS:
            if provider != settings.provider and get_api_key(provider):
                specs.append((provider, DEFAULT_JUDGE_MODELS[provider]))

    seen = set()
    unique_specs = []
    for spec in specs:
        if spec not in seen and spec[0] in SUPPORTED_PROVIDERS:
            seen.add(spec)
            unique_specs.append(spec)
    return unique_specs


def create_judge() -> DeepEvalBaseLLM:
    """Build the configured judge chain or raise ``JudgeUnavailableError``."""
    settings = JudgeSettings.from_env()

    if settings.provider == "mock":
        raise JudgeUnavailableError(
            "Mock judges are intentionally disabled: they hide real regressions. "
            "Configure GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY."
        )

    judges: list[DeepEvalBaseLLM] = []
    for provider, model in _judge_specs(settings):
        api_key = get_api_key(provider)
        if not api_key:
            continue
        judges.append(
            OpenAICompatibleJudge(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=OPENAI_COMPATIBLE_BASE_URLS[provider],
            )
        )

    if not judges:
        raise JudgeUnavailableError(
            "No judge credentials found. Set GEMINI_API_KEY (recommended), "
            "GROQ_API_KEY, or OPENAI_API_KEY in the environment or .env file."
        )
    _mark_judge_success(judges[0])
    if len(judges) == 1:
        return judges[0]
    return FallbackJudge(judges)
