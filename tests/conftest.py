"""Shared pytest configuration: paths, fixtures and metric logging."""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("PROMPTFOO_DISABLE_TELEMETRY", "1")

from src.agent import SupportAgent  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"


def _worker_suffix() -> str:
    worker = os.getenv("PYTEST_XDIST_WORKER", "").strip()
    return f"_{worker}" if worker else ""


def metrics_log_path() -> Path:
    return RESULTS_DIR / f"deepeval_metrics{_worker_suffix()}.json"


def _clean_metric_logs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in glob.glob(str(RESULTS_DIR / "deepeval_metrics*.json")):
        try:
            os.remove(path)
        except OSError:
            pass


def _append_metric(record: dict[str, Any]) -> None:
    path = metrics_log_path()
    records: list[dict[str, Any]] = []
    if path.exists():
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            records = []

    record.setdefault(
        "timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    records.append(record)

    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_path, path)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "llm: requires a live LLM judge and consumes API quota"
    )
    config.addinivalue_line("markers", "fast: deterministic test with no network calls")


def pytest_sessionstart(session: pytest.Session) -> None:
    # The one-click runner and CI split deterministic and LLM phases into
    # separate pytest processes. Set EVAL_KEEP_METRICS=1 for the later phase so
    # the deterministic ToolSelectionAccuracy records survive. Only the
    # controller should clear logs when running under xdist.
    keep_existing = os.getenv("EVAL_KEEP_METRICS", "0").lower() in {"1", "true", "yes"}
    if not os.getenv("PYTEST_XDIST_WORKER") and not keep_existing:
        _clean_metric_logs()


@pytest.fixture(scope="session")
def golden_cases() -> list[dict[str, Any]]:
    dataset_path = PROJECT_ROOT / "data" / "golden_dataset.json"
    return json.loads(dataset_path.read_text(encoding="utf-8"))


@pytest.fixture()
def agent() -> SupportAgent:
    instance = SupportAgent(mode="deterministic")
    yield instance
    instance.reset()


@pytest.fixture(scope="session")
def eval_judge():
    # Imported lazily so deterministic tests do not pay DeepEval's startup cost.
    from tests.judge import JudgeUnavailableError, create_judge

    try:
        return create_judge()
    except JudgeUnavailableError as exc:
        pytest.skip(str(exc))


def _judge_model_label() -> str:
    """Prefer the judge model that actually answered over static config."""
    try:
        from tests.judge import effective_judge_model
    except Exception:
        return os.getenv("EVAL_JUDGE_MODEL", "").strip() or "unknown"
    return (
        effective_judge_model()
        or os.getenv("EVAL_JUDGE_MODEL", "").strip()
        or "unknown"
    )


@pytest.fixture()
def record_metric() -> Callable[..., None]:
    def _record(
        *,
        test_name: str,
        metric: str,
        score: float,
        threshold: float,
        passed: bool,
        reason: str = "",
        error: str | None = None,
    ) -> None:
        _append_metric(
            {
                "test_name": test_name,
                "metric": metric,
                "score": round(float(score), 4),
                "threshold": threshold,
                "passed": bool(passed),
                "reason": reason,
                "error": error,
                "model": _judge_model_label(),
            }
        )

    return _record
