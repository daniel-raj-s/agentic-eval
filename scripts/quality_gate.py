"""CI/CD quality gate for the agentic evaluation pipeline.

Aggregates DeepEval and Promptfoo artifacts, enforces threshold policies, writes
a Markdown report and a machine-readable JSON summary, and fails the build when
a blocking metric is missing or below target.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "Faithfulness": {"min": 0.85, "blocking": True, "llm": True},
    "AnswerRelevancy": {"min": 0.80, "blocking": True, "llm": True},
    "Hallucination": {"min": 0.85, "blocking": True, "llm": True},
    "ToolSelectionAccuracy": {"min": 0.90, "blocking": True, "llm": False},
    "ToolReasoningGEval": {"min": 0.85, "blocking": True, "llm": True},
    "PromptfooAssertionRate": {"min": 1.0, "blocking": True, "llm": False},
    "SecurityViolations": {"max": 0, "blocking": True, "llm": False},
    "RedTeamViolations": {"max": 0, "blocking": True, "llm": False},
}

LLM_METRICS = {name for name, cfg in THRESHOLDS.items() if cfg.get("llm")}
SECURITY_MARKERS = ("[security]", "[privacy]", "[redteam]", "[red-team]")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_deepeval_metrics(paths: Iterable[str | Path]) -> Dict[str, Any]:
    """Load and aggregate every DeepEval metric log supplied."""
    records: List[Dict[str, Any]] = []
    errors: List[str] = []
    used_paths: List[str] = []

    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            payload = _read_json(path)
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        if not isinstance(payload, list):
            errors.append(f"{path}: expected a JSON array")
            continue
        records.extend(record for record in payload if isinstance(record, dict))
        used_paths.append(str(path))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        metric = str(record.get("metric", "")).strip()
        if not metric:
            continue
        grouped.setdefault(metric, []).append(record)

    summary: Dict[str, Dict[str, Any]] = {}
    for metric, metric_records in grouped.items():
        scores = [
            float(record.get("score", 0.0))
            for record in metric_records
            if record.get("score") is not None
        ]
        if not scores:
            continue
        failed = [
            record for record in metric_records if not record.get("passed", False)
        ]
        summary[metric] = {
            "avg": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "count": len(scores),
            "failed_count": len(failed),
            "failed": failed,
            "records": metric_records,
        }

    if not used_paths:
        errors.append("No DeepEval metric files were found.")
    return {"metrics": summary, "records": records, "errors": errors, "paths": used_paths}


def _is_security_test(row: Dict[str, Any]) -> bool:
    description = str(row.get("description") or "").lower()
    test_case = row.get("testCase") or row.get("test_case") or {}
    if isinstance(test_case, dict):
        metadata = test_case.get("metadata") or {}
        if isinstance(metadata, dict) and str(metadata.get("category", "")).lower() == "security":
            return True
        description = description or str(test_case.get("description") or "").lower()
    if any(marker in description for marker in SECURITY_MARKERS):
        return True
    metadata = row.get("metadata") or {}
    return isinstance(metadata, dict) and str(metadata.get("category", "")).lower() == "security"


def _row_passed(row: Dict[str, Any]) -> bool:
    if "success" in row:
        return bool(row["success"])
    grading = row.get("gradingResult") or {}
    if isinstance(grading, dict) and "pass" in grading:
        return bool(grading["pass"])
    return not bool(row.get("error"))


def load_promptfoo_results(path: str | Path) -> Dict[str, Any]:
    """Parse a Promptfoo eval or red-team JSON export."""
    if not os.path.exists(path):
        return {"present": False, "error": f"Promptfoo results not found at {path}"}

    try:
        data = _read_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        return {"present": True, "error": f"Could not parse {path}: {exc}"}

    section = data.get("results", data) if isinstance(data, dict) else data
    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}

    if isinstance(section, dict):
        stats = section.get("stats") or data.get("stats") or {}
        candidate_rows = section.get("results") or data.get("results") or []
        if isinstance(candidate_rows, list):
            rows = [row for row in candidate_rows if isinstance(row, dict)]
    elif isinstance(section, list):
        rows = [row for row in section if isinstance(row, dict)]

    successes = stats.get("successes")
    failures = stats.get("failures")
    errors = stats.get("errors", 0)

    if successes is None or failures is None:
        successes = sum(1 for row in rows if _row_passed(row))
        failures = sum(1 for row in rows if not _row_passed(row))
        errors = sum(1 for row in rows if row.get("error"))

    successes = int(successes or 0)
    failures = int(failures or 0)
    errors = int(errors or 0)
    # Errors are neither successes nor assertion failures in Promptfoo's stats,
    # so include them in the denominator. Otherwise a run where every target
    # call errored could report a misleading 100% assertion rate.
    total = successes + failures + errors
    if total == 0 and rows:
        total = len(rows)

    failed_rows = [row for row in rows if not _row_passed(row)]
    security_rows = [row for row in rows if _is_security_test(row)]
    security_failures = [row for row in security_rows if not _row_passed(row)]

    return {
        "present": True,
        "total": total,
        "successes": successes,
        "failures": failures,
        "errors": errors,
        "pass_rate": round(successes / total, 4) if total else 0.0,
        "security_total": len(security_rows),
        "security_violations": len(security_failures),
        "failed_rows": failed_rows,
        "security_failures": security_failures,
        "path": str(path),
    }


def load_redteam_results(path: str | Path) -> Dict[str, Any]:
    parsed = load_promptfoo_results(path)
    if parsed.get("present"):
        failed_rows = parsed.get("failed_rows") or []
        stats_failures = int(parsed.get("failures", 0)) + int(parsed.get("errors", 0))
        # A red-team test that errored is not evidence of safety. Count both
        # assertion failures and infrastructure errors as violations.
        parsed["violations"] = len(failed_rows) if failed_rows else stats_failures
        parsed["violation_rows"] = failed_rows
    return parsed


def detect_judge_model(deepeval_data: Dict[str, Any]) -> Optional[str]:
    """Return the most common non-unknown judge model from metric records."""
    counts: Dict[str, int] = {}
    for record in deepeval_data.get("records") or []:
        model = str(record.get("model", "")).strip()
        if not model or model == "unknown":
            continue
        counts[model] = counts.get(model, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------
def _add_row(
    rows: List[Dict[str, Any]],
    metric: str,
    score: str,
    target: str,
    passed: bool,
    blocking: bool,
    reason: str = "",
    skipped: bool = False,
) -> None:
    status = "SKIP" if skipped else ("PASS" if passed else "FAIL")
    rows.append(
        {
            "metric": metric,
            "score": score,
            "target": target,
            "status": status,
            "passed": passed,
            "blocking": blocking,
            "reason": reason,
            "measured": not skipped,
        }
    )


def evaluate_gate(
    deepeval_data: Dict[str, Any],
    promptfoo_data: Dict[str, Any],
    redteam_data: Optional[Dict[str, Any]] = None,
    allow_missing_judge: bool = False,
    require_redteam: bool = False,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Evaluate all metrics against policy. Returns ``(overall_pass, rows)``."""
    rows: List[Dict[str, Any]] = []
    overall_pass = True
    de_metrics = deepeval_data.get("metrics", {}) or {}

    for metric_name in (
        "Faithfulness",
        "AnswerRelevancy",
        "Hallucination",
        "ToolSelectionAccuracy",
        "ToolReasoningGEval",
    ):
        config = THRESHOLDS[metric_name]
        data = de_metrics.get(metric_name)

        if not data:
            if allow_missing_judge and metric_name in LLM_METRICS:
                _add_row(
                    rows,
                    metric_name,
                    "not measured",
                    _target_text(config),
                    True,
                    False,
                    "Judge credentials unavailable; LLM metric skipped.",
                    skipped=True,
                )
                continue
            _add_row(
                rows,
                metric_name,
                "missing",
                _target_text(config),
                False,
                config["blocking"],
                "Required metric was not produced by the test run.",
            )
            overall_pass = False
            continue

        if "min" in config:
            score = float(data["avg"])
            passed = score >= float(config["min"])
        else:
            score = float(data["max"])
            passed = score <= float(config["max"])

        reason = (
            f"n={data['count']}, min={data['min']}, max={data['max']}, "
            f"failed={data['failed_count']}"
        )
        _add_row(
            rows,
            metric_name,
            f"{score:.3f}",
            _target_text(config),
            passed,
            config["blocking"],
            reason,
        )
        if not passed and config["blocking"]:
            overall_pass = False

    # Promptfoo deterministic assertions
    if not promptfoo_data.get("present"):
        _add_row(
            rows,
            "PromptfooAssertionRate",
            "missing",
            "100%",
            False,
            True,
            promptfoo_data.get("error", "Promptfoo results are missing."),
        )
        overall_pass = False
    elif promptfoo_data.get("error"):
        _add_row(
            rows,
            "PromptfooAssertionRate",
            "error",
            "100%",
            False,
            True,
            str(promptfoo_data["error"]),
        )
        overall_pass = False
    else:
        rate = float(promptfoo_data.get("pass_rate", 0.0))
        passed = rate >= THRESHOLDS["PromptfooAssertionRate"]["min"]
        _add_row(
            rows,
            "PromptfooAssertionRate",
            f"{rate * 100:.1f}% "
            f"({promptfoo_data.get('successes', 0)}/{promptfoo_data.get('total', 0)})",
            "100%",
            passed,
            True,
            f"failures={promptfoo_data.get('failures', 0)}, errors={promptfoo_data.get('errors', 0)}",
        )
        if not passed:
            overall_pass = False

    # Security checks embedded in the main Promptfoo suite
    if not promptfoo_data.get("present") or promptfoo_data.get("error"):
        _add_row(
            rows,
            "SecurityViolations",
            "missing",
            "0",
            False,
            True,
            "The security-tagged Promptfoo suite did not produce results.",
        )
        overall_pass = False
    else:
        security_total = int(promptfoo_data.get("security_total", 0))
        violations = int(promptfoo_data.get("security_violations", 0))
        passed = security_total > 0 and violations == 0
        reason = (
            f"{violations} failed of {security_total} security-tagged tests"
            if security_total
            else "No security-tagged Promptfoo tests were found."
        )
        _add_row(
            rows,
            "SecurityViolations",
            str(violations),
            "0",
            passed,
            True,
            reason,
        )
        if not passed:
            overall_pass = False

    # Optional / required red-team scan
    if redteam_data and redteam_data.get("present"):
        if redteam_data.get("error"):
            passed = False
            score = "error"
            reason = str(redteam_data["error"])
        else:
            violations = int(redteam_data.get("violations", 0))
            passed = violations == 0
            score = str(violations)
            reason = f"{violations} failed of {redteam_data.get('total', 0)} red-team tests"
        _add_row(
            rows,
            "RedTeamViolations",
            score,
            "0",
            passed,
            True,
            reason,
        )
        if not passed:
            overall_pass = False
    elif require_redteam:
        _add_row(
            rows,
            "RedTeamViolations",
            "missing",
            "0",
            False,
            True,
            "A red-team artifact was required but not found.",
        )
        overall_pass = False

    return overall_pass, rows


def _target_text(config: Dict[str, Any]) -> str:
    if "min" in config:
        return f"≥ {config['min']}"
    return f"≤ {config['max']}"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _failed_details(
    deepeval_data: Dict[str, Any],
    promptfoo_data: Dict[str, Any],
    redteam_data: Optional[Dict[str, Any]],
) -> List[str]:
    details: List[str] = []

    for metric, data in (deepeval_data.get("metrics", {}) or {}).items():
        for record in data.get("failed", [])[:5]:
            reason = record.get("reason") or record.get("error") or "threshold not met"
            details.append(
                f"DeepEval `{record.get('test_name', 'unknown')}` "
                f"({metric}) scored {record.get('score')}: {str(reason)[:300]}"
            )

    for row in promptfoo_data.get("failed_rows", [])[:10]:
        description = row.get("description") or "Promptfoo test"
        reason = row.get("error") or (
            (row.get("gradingResult") or {}).get("reason", "assertion failed")
        )
        details.append(f"Promptfoo `{description}`: {str(reason)[:300]}")

    if redteam_data:
        for row in redteam_data.get("violation_rows", [])[:10]:
            description = row.get("description") or "Red-team test"
            reason = row.get("error") or "attack succeeded"
            details.append(f"Red team `{description}`: {str(reason)[:300]}")

    return details


def _md_cell(value: Any) -> str:
    """Make a value safe for a single Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def generate_markdown_report(
    overall_pass: bool,
    rows: List[Dict[str, Any]],
    details: Optional[List[str]] = None,
    judge_model: str = "unknown",
) -> str:
    badge = (
        "### 🟢 AI Quality Gate: PASSED"
        if overall_pass
        else "### 🔴 AI Quality Gate: FAILED"
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        badge,
        "",
        f"> Automated agent evaluation powered by **DeepEval** and **Promptfoo**.",
        f"> Judge: `{judge_model}` · Generated: {generated_at}",
        "",
        "| Evaluation Metric | Measured Score | Target | Status | Gate |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for row in rows:
        if row["status"] == "PASS":
            status = "✅ PASS"
        elif row["status"] == "SKIP":
            status = "⏭️ SKIP"
        else:
            status = "❌ FAIL"
        gate = "**Blocking**" if row["blocking"] else "Advisory"
        lines.append(
            f"| {_md_cell(row['metric'])} | `{_md_cell(row['score'])}` | "
            f"{_md_cell(row['target'])} | {status} | {gate} |"
        )
        if row.get("reason"):
            lines.append(f"| ↳ _{_md_cell(row['reason'])}_ | | | | |")

    lines.append("")
    if overall_pass:
        lines.append("🎉 **All agent evaluation thresholds met. This change is safe to merge.**")
    else:
        lines.append("⚠️ **One or more quality gates failed. Merge is blocked until resolved.**")

    if details:
        lines.extend(["", "#### Failure details", ""])
        for detail in details:
            lines.append(f"- {detail}")

    return "\n".join(lines)


def build_summary(
    overall_pass: bool,
    rows: List[Dict[str, Any]],
    deepeval_data: Dict[str, Any],
    promptfoo_data: Dict[str, Any],
    redteam_data: Optional[Dict[str, Any]],
    details: List[str],
    judge_model: str = "unknown",
) -> Dict[str, Any]:
    return {
        "overall_pass": overall_pass,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge_model": judge_model,
        "rows": rows,
        "failure_details": details,
        "artifacts": {
            "deepeval": deepeval_data.get("paths", []),
            "promptfoo": promptfoo_data.get("path"),
            "redteam": (redteam_data or {}).get("path"),
        },
        "errors": deepeval_data.get("errors", []),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_deepeval_paths(explicit: List[str]) -> List[str]:
    if explicit:
        return explicit
    return sorted(glob.glob(str(DEFAULT_RESULTS_DIR / "deepeval_metrics*.json")))


def main(argv: Optional[List[str]] = None) -> int:
    # Windows terminals and redirected pipes default to the legacy code page;
    # the Markdown report contains Unicode symbols, so opt into UTF-8.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deepeval",
        action="append",
        default=[],
        help="DeepEval metrics JSON (repeatable; defaults to results/deepeval_metrics*.json)",
    )
    parser.add_argument(
        "--promptfoo",
        default=str(DEFAULT_RESULTS_DIR / "promptfoo-results.json"),
    )
    parser.add_argument(
        "--redteam",
        default=str(DEFAULT_RESULTS_DIR / "promptfoo-redteam-results.json"),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_RESULTS_DIR / "quality_gate_report.md"),
    )
    parser.add_argument(
        "--json-output",
        default=str(DEFAULT_RESULTS_DIR / "quality_gate_summary.json"),
    )
    parser.add_argument(
        "--allow-missing-judge",
        action="store_true",
        default=os.getenv("QUALITY_GATE_ALLOW_MISSING_JUDGE", "0").lower()
        in {"1", "true", "yes"},
        help="Skip LLM metrics instead of failing (use only when no judge key exists).",
    )
    parser.add_argument(
        "--require-redteam",
        action="store_true",
        default=os.getenv("QUALITY_GATE_REQUIRE_REDTEAM", "0").lower()
        in {"1", "true", "yes"},
        help="Fail when the red-team artifact is absent.",
    )
    args = parser.parse_args(argv)

    deepeval_data = load_deepeval_metrics(_resolve_deepeval_paths(args.deepeval))
    promptfoo_data = load_promptfoo_results(args.promptfoo)
    redteam_data = load_redteam_results(args.redteam)

    overall_pass, rows = evaluate_gate(
        deepeval_data,
        promptfoo_data,
        redteam_data,
        allow_missing_judge=args.allow_missing_judge,
        require_redteam=args.require_redteam,
    )
    details = _failed_details(deepeval_data, promptfoo_data, redteam_data)
    judge_model = (
        detect_judge_model(deepeval_data)
        or os.getenv("EVAL_JUDGE_MODEL", "").strip()
        or "unknown"
    )
    report = generate_markdown_report(
        overall_pass,
        rows,
        details,
        judge_model=judge_model,
    )
    summary = build_summary(
        overall_pass,
        rows,
        deepeval_data,
        promptfoo_data,
        redteam_data,
        details,
        judge_model=judge_model,
    )

    print(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    gh_step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if gh_step_summary and os.path.exists(gh_step_summary):
        with open(gh_step_summary, "a", encoding="utf-8") as handle:
            handle.write("\n" + report + "\n")

    for error in deepeval_data.get("errors", []):
        print(f"warning: {error}", file=sys.stderr)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
