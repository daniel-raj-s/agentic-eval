"""Unit tests for the CI quality gate.

These tests are deterministic and never call a model. They verify the policy
that turns evaluation artifacts into a merge decision.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from scripts import quality_gate

DETERMINISTIC_ONLY = {
    "ToolSelectionAccuracy": 1.0,
}

LLM_DEFAULTS = {
    "Faithfulness": 0.95,
    "AnswerRelevancy": 0.93,
    "Hallucination": 0.95,
    "ToolReasoningGEval": 0.90,
}


def _metric_entry(score: float, *, metric: str = "Metric") -> Dict[str, Any]:
    return {
        "avg": round(score, 4),
        "min": round(score, 4),
        "max": round(score, 4),
        "count": 1,
        "failed_count": 0,
        "failed": [],
        "records": [{"metric": metric, "score": score, "passed": True}],
    }


def _deepeval_data(scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    values = {**LLM_DEFAULTS, **DETERMINISTIC_ONLY}
    if scores is not None:
        values = scores
    return {
        "metrics": {
            name: _metric_entry(score, metric=name) for name, score in values.items()
        },
        "records": [],
        "errors": [],
        "paths": ["results/deepeval_metrics.json"],
    }


def _promptfoo_data(**overrides: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "present": True,
        "total": 10,
        "successes": 10,
        "failures": 0,
        "errors": 0,
        "pass_rate": 1.0,
        "security_total": 3,
        "security_violations": 0,
        "failed_rows": [],
        "security_failures": [],
        "path": "results/promptfoo-results.json",
    }
    data.update(overrides)
    return data


def _redteam_data(**overrides: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "present": True,
        "total": 4,
        "successes": 4,
        "failures": 0,
        "errors": 0,
        "pass_rate": 1.0,
        "violations": 0,
        "violation_rows": [],
        "path": "results/promptfoo-redteam-results.json",
    }
    data.update(overrides)
    return data


def _row(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    return next(row for row in rows if row["metric"] == metric)


def test_all_thresholds_met_passes() -> None:
    passed, rows = quality_gate.evaluate_gate(
        _deepeval_data(), _promptfoo_data(), _redteam_data()
    )
    assert passed is True
    assert {row["metric"] for row in rows} == {
        "Faithfulness",
        "AnswerRelevancy",
        "Hallucination",
        "ToolSelectionAccuracy",
        "ToolReasoningGEval",
        "PromptfooAssertionRate",
        "SecurityViolations",
        "RedTeamViolations",
    }
    assert all(row["status"] == "PASS" for row in rows)


def test_missing_metrics_fail_by_default() -> None:
    passed, rows = quality_gate.evaluate_gate(
        _deepeval_data(DETERMINISTIC_ONLY), _promptfoo_data()
    )
    assert passed is False
    for metric in ("Faithfulness", "AnswerRelevancy", "Hallucination", "ToolReasoningGEval"):
        assert _row(rows, metric)["status"] == "FAIL"
        assert _row(rows, metric)["score"] == "missing"


def test_allow_missing_judge_skips_llm_metrics_only() -> None:
    passed, rows = quality_gate.evaluate_gate(
        _deepeval_data(DETERMINISTIC_ONLY),
        _promptfoo_data(),
        allow_missing_judge=True,
    )
    assert passed is True
    for metric in LLM_DEFAULTS:
        assert _row(rows, metric)["status"] == "SKIP"
        assert _row(rows, metric)["measured"] is False
    assert _row(rows, "ToolSelectionAccuracy")["status"] == "PASS"


def test_missing_deterministic_metric_still_fails_with_skip_flag() -> None:
    passed, rows = quality_gate.evaluate_gate(
        _deepeval_data({"Faithfulness": 0.95}),
        _promptfoo_data(),
        allow_missing_judge=True,
    )
    assert passed is False
    assert _row(rows, "ToolSelectionAccuracy")["status"] == "FAIL"


def test_low_faithfulness_fails() -> None:
    scores = {**LLM_DEFAULTS, **DETERMINISTIC_ONLY, "Faithfulness": 0.70}
    passed, rows = quality_gate.evaluate_gate(_deepeval_data(scores), _promptfoo_data())
    assert passed is False
    assert _row(rows, "Faithfulness")["status"] == "FAIL"


def test_low_hallucination_score_fails() -> None:
    scores = {**LLM_DEFAULTS, **DETERMINISTIC_ONLY, "Hallucination": 0.40}
    passed, rows = quality_gate.evaluate_gate(_deepeval_data(scores), _promptfoo_data())
    assert passed is False
    assert _row(rows, "Hallucination")["status"] == "FAIL"


def test_promptfoo_assertion_failure_blocks_merge() -> None:
    passed, rows = quality_gate.evaluate_gate(
        _deepeval_data(),
        _promptfoo_data(
            successes=9,
            failures=1,
            pass_rate=0.9,
            failed_rows=[{"description": "broken case", "error": "assertion failed"}],
        ),
    )
    assert passed is False
    assert _row(rows, "PromptfooAssertionRate")["status"] == "FAIL"


def test_missing_promptfoo_artifact_fails_both_promptfoo_gates() -> None:
    passed, rows = quality_gate.evaluate_gate(
        _deepeval_data(),
        {"present": False, "error": "not found"},
    )
    assert passed is False
    assert _row(rows, "PromptfooAssertionRate")["status"] == "FAIL"
    assert _row(rows, "SecurityViolations")["status"] == "FAIL"


def test_security_violation_blocks_merge() -> None:
    passed, rows = quality_gate.evaluate_gate(
        _deepeval_data(),
        _promptfoo_data(
            security_total=3,
            security_violations=1,
            security_failures=[{"description": "injection"}],
        ),
    )
    assert passed is False
    assert _row(rows, "SecurityViolations")["status"] == "FAIL"


def test_missing_security_tests_fails_closed() -> None:
    passed, rows = quality_gate.evaluate_gate(
        _deepeval_data(),
        _promptfoo_data(security_total=0, security_violations=0),
    )
    assert passed is False
    assert _row(rows, "SecurityViolations")["status"] == "FAIL"


def test_required_redteam_artifact_must_exist() -> None:
    passed, rows = quality_gate.evaluate_gate(
        _deepeval_data(),
        _promptfoo_data(),
        redteam_data={"present": False},
        require_redteam=True,
    )
    assert passed is False
    assert _row(rows, "RedTeamViolations")["status"] == "FAIL"


def test_redteam_violation_blocks_merge() -> None:
    passed, rows = quality_gate.evaluate_gate(
        _deepeval_data(),
        _promptfoo_data(),
        _redteam_data(violations=2, failures=2),
    )
    assert passed is False
    assert _row(rows, "RedTeamViolations")["status"] == "FAIL"


def test_detect_judge_model_prefers_most_common_actual_model() -> None:
    data = {
        "records": [
            {"metric": "Faithfulness", "model": "unknown"},
            {"metric": "Faithfulness", "model": "groq:openai/gpt-oss-120b"},
            {"metric": "AnswerRelevancy", "model": "groq:openai/gpt-oss-120b"},
            {"metric": "Hallucination", "model": "gemini:gemini-3.8-flash"},
        ]
    }
    assert quality_gate.detect_judge_model(data) == "groq:openai/gpt-oss-120b"
    assert quality_gate.detect_judge_model({"records": [{"model": "unknown"}]}) is None


def test_load_deepeval_metrics_aggregates_scores(tmp_path: Path) -> None:
    path = tmp_path / "deepeval_metrics.json"
    path.write_text(
        json.dumps(
            [
                {"metric": "Faithfulness", "score": 0.8, "passed": True},
                {"metric": "Faithfulness", "score": 1.0, "passed": True},
                {
                    "metric": "Hallucination",
                    "score": 0.4,
                    "passed": False,
                    "reason": "invented fee",
                },
            ]
        ),
        encoding="utf-8",
    )

    data = quality_gate.load_deepeval_metrics([path])
    assert data["errors"] == []
    assert data["metrics"]["Faithfulness"]["avg"] == 0.9
    assert data["metrics"]["Faithfulness"]["count"] == 2
    assert data["metrics"]["Hallucination"]["failed_count"] == 1


def test_load_promptfoo_results_reads_wrapped_export(tmp_path: Path) -> None:
    path = tmp_path / "promptfoo-results.json"
    path.write_text(
        json.dumps(
            {
                "results": {
                    "stats": {"successes": 1, "failures": 1, "errors": 0},
                    "results": [
                        {
                            "success": True,
                            "testCase": {
                                "description": "TC-1 [security] injection",
                                "metadata": {"category": "security"},
                            },
                        },
                        {
                            "success": False,
                            "error": "assertion failed",
                            "testCase": {"description": "TC-2 functional"},
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    data = quality_gate.load_promptfoo_results(path)
    assert data["present"] is True
    assert data["total"] == 2
    assert data["successes"] == 1
    assert data["failures"] == 1
    assert data["security_total"] == 1
    assert data["security_violations"] == 0
    assert data["pass_rate"] == 0.5


def test_load_promptfoo_results_counts_errors_in_pass_rate(tmp_path: Path) -> None:
    path = tmp_path / "promptfoo-results.json"
    path.write_text(
        json.dumps(
            {
                "results": {
                    "stats": {"successes": 1, "failures": 0, "errors": 1},
                    "results": [
                        {"success": True, "testCase": {"description": "ok"}},
                        {
                            "success": False,
                            "error": "provider crashed",
                            "testCase": {"description": "errored"},
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    data = quality_gate.load_promptfoo_results(path)
    assert data["total"] == 2
    assert data["errors"] == 1
    assert data["pass_rate"] == 0.5
    assert len(data["failed_rows"]) == 1


def test_load_redteam_results_counts_errors_as_violations(tmp_path: Path) -> None:
    path = tmp_path / "promptfoo-redteam-results.json"
    path.write_text(
        json.dumps(
            {
                "results": {
                    "stats": {"successes": 1, "failures": 0, "errors": 1},
                    "results": [
                        {"success": True, "testCase": {"description": "blocked attack"}},
                        {
                            "success": False,
                            "error": "grader unavailable",
                            "testCase": {"description": "errored attack"},
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    data = quality_gate.load_redteam_results(path)
    assert data["total"] == 2
    assert data["violations"] == 1
    assert len(data["violation_rows"]) == 1


def test_markdown_report_escapes_table_cells() -> None:
    report = quality_gate.generate_markdown_report(
        False,
        [
            {
                "metric": "Odd|Metric",
                "score": "1|2",
                "target": "0",
                "status": "FAIL",
                "blocking": True,
                "reason": "line one\nline two",
            }
        ],
        details=["failure details"],
        judge_model="gemini:test",
    )
    assert "Odd\\|Metric" in report
    assert "1\\|2" in report
    assert "line one line two" in report
    assert "AI Quality Gate: FAILED" in report


def test_cli_writes_reports_and_returns_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    deepeval_path = tmp_path / "deepeval_metrics.json"
    deepeval_path.write_text(
        json.dumps(
            [
                {"metric": "Faithfulness", "score": 0.95, "passed": True},
                {"metric": "AnswerRelevancy", "score": 0.92, "passed": True},
                {"metric": "Hallucination", "score": 0.95, "passed": True},
                {"metric": "ToolSelectionAccuracy", "score": 1.0, "passed": True},
                {"metric": "ToolReasoningGEval", "score": 0.9, "passed": True},
            ]
        ),
        encoding="utf-8",
    )
    promptfoo_path = tmp_path / "promptfoo-results.json"
    promptfoo_path.write_text(
        json.dumps(
            {
                "results": {
                    "stats": {"successes": 2, "failures": 0, "errors": 0},
                    "results": [
                        {
                            "success": True,
                            "testCase": {
                                "description": "security",
                                "metadata": {"category": "security"},
                            },
                        },
                        {"success": True, "testCase": {"description": "functional"}},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.md"
    summary_path = tmp_path / "summary.json"
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    code = quality_gate.main(
        [
            "--deepeval",
            str(deepeval_path),
            "--promptfoo",
            str(promptfoo_path),
            "--output",
            str(report_path),
            "--json-output",
            str(summary_path),
            "--redteam",
            str(tmp_path / "redteam-results.json"),
        ]
    )

    assert code == 0
    assert "AI Quality Gate: PASSED" in capsys.readouterr().out
    assert report_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["overall_pass"] is True
    assert summary["artifacts"]["deepeval"] == [str(deepeval_path)]


def test_cli_returns_failure_when_promptfoo_artifact_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deepeval_path = tmp_path / "deepeval_metrics.json"
    deepeval_path.write_text(
        json.dumps(
            [
                {"metric": name, "score": score, "passed": True}
                for name, score in {
                    **LLM_DEFAULTS,
                    **DETERMINISTIC_ONLY,
                }.items()
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    code = quality_gate.main(
        [
            "--deepeval",
            str(deepeval_path),
            "--promptfoo",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "report.md"),
            "--json-output",
            str(tmp_path / "summary.json"),
            "--redteam",
            str(tmp_path / "redteam-results.json"),
        ]
    )

    assert code == 1
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall_pass"] is False


def test_cli_allow_missing_judge_exits_zero_without_llm_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deepeval_path = tmp_path / "deepeval_metrics.json"
    deepeval_path.write_text(
        json.dumps(
            [{"metric": "ToolSelectionAccuracy", "score": 1.0, "passed": True}]
        ),
        encoding="utf-8",
    )
    promptfoo_path = tmp_path / "promptfoo-results.json"
    promptfoo_path.write_text(
        json.dumps(
            {
                "results": {
                    "stats": {"successes": 1, "failures": 0, "errors": 0},
                    "results": [
                        {
                            "success": True,
                            "testCase": {
                                "description": "security",
                                "metadata": {"category": "security"},
                            },
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    code = quality_gate.main(
        [
            "--deepeval",
            str(deepeval_path),
            "--promptfoo",
            str(promptfoo_path),
            "--allow-missing-judge",
            "--redteam",
            str(tmp_path / "redteam-results.json"),
            "--output",
            str(tmp_path / "report.md"),
            "--json-output",
            str(tmp_path / "summary.json"),
        ]
    )

    assert code == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    skipped = [row for row in summary["rows"] if row["status"] == "SKIP"]
    assert {row["metric"] for row in skipped} == set(LLM_DEFAULTS)
