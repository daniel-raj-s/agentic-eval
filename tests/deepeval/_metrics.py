"""Shared helper for measuring a DeepEval metric and persisting its result."""
from __future__ import annotations

from typing import Any

import pytest
from deepeval.test_case import LLMTestCase


def measure_and_record(
    metric: Any,
    test_case: LLMTestCase,
    record_metric,
    test_name: str,
    metric_name: str | None = None,
) -> None:
    """Measure once, persist the score, then fail with a useful message."""
    resolved_name = metric_name or metric.__class__.__name__.replace("Metric", "")
    error = None
    try:
        metric.measure(test_case)
    except Exception as exc:  # judge, schema or network failure
        error = f"{type(exc).__name__}: {exc}"
        record_metric(
            test_name=test_name,
            metric=resolved_name,
            score=0.0,
            threshold=float(metric.threshold),
            passed=False,
            reason="Metric execution failed.",
            error=error,
        )
        pytest.fail(f"{test_name} could not be measured: {error}")

    score = float(metric.score or 0.0)
    passed = bool(metric.is_successful())
    reason = getattr(metric, "reason", "") or ""
    record_metric(
        test_name=test_name,
        metric=resolved_name,
        score=score,
        threshold=float(metric.threshold),
        passed=passed,
        reason=reason,
        error=error,
    )
    assert passed, (
        f"{test_name} failed: score={score:.3f}, threshold={metric.threshold}, "
        f"reason={reason}"
    )
