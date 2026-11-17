"""Deterministic tool-selection and tool-contract tests.

These run without a judge, record the ``ToolSelectionAccuracy`` metric used by
the quality gate, and stay outside the DeepEval directory so a free CI run does
not import the semantic-metric stack at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_CASES = json.loads(
    (PROJECT_ROOT / "data" / "golden_dataset.json").read_text(encoding="utf-8")
)
TOOL_CASES = [case for case in GOLDEN_CASES if "expected_tool" in case]


@pytest.mark.fast
@pytest.mark.parametrize("case", TOOL_CASES, ids=lambda case: case["id"])
def test_agent_tool_selection_deterministic(case, agent, record_metric) -> None:
    """The required tool must be invoked with the expected arguments."""
    response = agent.run(case["query"])
    called_tools = {tool.name: tool.args for tool in response.tools_called}
    expected_tool = case["expected_tool"]

    try:
        assert expected_tool in called_tools, (
            f"Expected tool '{expected_tool}' was not invoked. "
            f"Invoked tools: {sorted(called_tools)}"
        )
        actual_args = called_tools[expected_tool]
        for key, expected_value in case.get("expected_tool_args", {}).items():
            assert actual_args.get(key) == expected_value, (
                f"Tool argument mismatch for '{key}': expected "
                f"{expected_value!r}, got {actual_args.get(key)!r}"
            )
    except AssertionError as exc:
        record_metric(
            test_name=f"{case['id']}_tool_selection",
            metric="ToolSelectionAccuracy",
            score=0.0,
            threshold=1.0,
            passed=False,
            reason=str(exc),
        )
        raise

    record_metric(
        test_name=f"{case['id']}_tool_selection",
        metric="ToolSelectionAccuracy",
        score=1.0,
        threshold=1.0,
        passed=True,
        reason=f"Correctly invoked {expected_tool} with matching arguments.",
    )


@pytest.mark.fast
def test_buyer_remorse_refund_math(agent) -> None:
    result = agent.tools.calculate_refund("ORD-1001", "buyer_remorse")
    assert result["eligible"] is True
    assert result["original_total"] == 149.99
    assert result["deductions"] == 5.99
    assert result["refund_amount"] == 144.00


@pytest.mark.fast
def test_return_request_has_no_shipping_fee(agent) -> None:
    result = agent.tools.calculate_refund("ORD-1001", "return_request")
    assert result["eligible"] is True
    assert result["deductions"] == 0.0
    assert result["refund_amount"] == 149.99


@pytest.mark.fast
def test_personalized_item_is_non_refundable(agent) -> None:
    result = agent.tools.calculate_refund("ORD-1004", "return_request")
    assert result["eligible"] is False
    assert "non-refundable" in result["reason"]
    assert result["refund_amount"] == 0.0


@pytest.mark.fast
def test_unknown_tool_raises(agent) -> None:
    with pytest.raises(ValueError, match="Unknown tool"):
        agent.tools.execute("drop_database", {})


@pytest.mark.fast
def test_tool_schemas_are_unique_and_complete(agent) -> None:
    names = [schema["name"] for schema in agent.tools.tool_schemas]
    assert len(names) == len(set(names)) == 4
    for schema in agent.tools.tool_schemas:
        assert schema["parameters"]["type"] == "object"
        assert schema["description"]
