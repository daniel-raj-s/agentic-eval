"""Promptfoo provider that wraps the e-commerce :class:`SupportAgent`.

The provider is deliberately stateless: every Promptfoo case gets a fresh agent
so a cancellation or refund in one test cannot leak into another. Responses
include tool calls, retrieval evidence, latency and guardrail decisions so the
YAML suite can assert on agent behaviour rather than only on prose.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bootstrap_paths() -> None:
    """Make the project and a local virtualenv importable by any interpreter."""
    candidates: List[Path] = [PROJECT_ROOT]
    venv = PROJECT_ROOT / ".venv"
    if venv.is_dir():
        candidates.append(venv / "Lib" / "site-packages")  # Windows
        candidates.extend(sorted(venv.glob("lib/python*/site-packages")))  # POSIX
    for candidate in candidates:
        text = str(candidate)
        if candidate.is_dir() and text not in sys.path:
            sys.path.append(text)


_bootstrap_paths()

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:  # pragma: no cover - dotenv ships with the project deps
    pass

from src.agent import SupportAgent  # noqa: E402


def _coerce_prompt(prompt: Any) -> str:
    """Extract the latest user message from string, chat or multi-input prompts."""
    if isinstance(prompt, dict):
        for key in ("query", "prompt", "__prompt", "message", "content"):
            value = prompt.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(prompt, ensure_ascii=False)

    text = str(prompt).strip()
    if not text:
        return text

    if text[0] in "[{":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(payload, list):
            for message in reversed(payload):
                if not isinstance(message, dict):
                    continue
                if message.get("role") not in {None, "user"}:
                    continue
                content = message.get("content") or message.get("text")
                if isinstance(content, str) and content.strip():
                    return content.strip()
            return text
        if isinstance(payload, dict):
            return _coerce_prompt(payload)
    return text


def _agent_config(options: Dict[str, Any]) -> Dict[str, Any]:
    config = options.get("config") or {}
    return config if isinstance(config, dict) else {}


def call_api(
    prompt: Any, options: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    """Promptfoo entry point. Must return ``output`` or ``error``."""
    config = _agent_config(options)
    query = _coerce_prompt(prompt)

    if not query:
        return {"error": "The provider received an empty prompt."}

    mode = str(config.get("mode") or os.getenv("AGENT_MODE", "deterministic"))
    provider = config.get("provider")
    model = config.get("model")

    try:
        agent = SupportAgent(mode=mode, provider=provider, model=model)
        response = agent.run(query, history=context.get("history") or None)
    except Exception as exc:  # configuration or infrastructure failure
        return {"error": f"{type(exc).__name__}: {exc}"}

    tools_called = [tool.model_dump() for tool in response.tools_called]
    metadata: Dict[str, Any] = {
        "mode": response.mode,
        "model": response.model,
        "blocked": response.blocked,
        "block_reason": response.block_reason,
        "latency_ms": response.latency_ms,
        "tools_called": tools_called,
        "tool_names": [tool.name for tool in response.tools_called],
        "retrieval_context": response.retrieval_context,
        "error": response.error,
    }

    redteam = (context.get("test") or {}).get("metadata") or {}
    if redteam:
        metadata["redteam"] = {
            "plugin_id": redteam.get("pluginId"),
            "strategy_id": redteam.get("strategyId"),
        }

    return {
        "output": response.output,
        "latencyMs": int(round(response.latency_ms)),
        "metadata": metadata,
        "guardrails": {
            "flagged": bool(response.blocked),
            "flaggedInput": bool(response.blocked),
            "flaggedOutput": False,
            "reason": response.block_reason,
        },
    }


if __name__ == "__main__":  # manual smoke test: python agent_provider.py "..."
    demo = sys.argv[1] if len(sys.argv) > 1 else "Where is my order ORD-1003?"
    print(json.dumps(call_api(demo, {"config": {}}, {"vars": {}}), indent=2))
