"""G-Eval tool-reasoning check (LLM-judged)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests.deepeval._metrics import measure_and_record

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_CASES = json.loads(
    (PROJECT_ROOT / "data" / "golden_dataset.json").read_text(encoding="utf-8")
)
REASONING_CASES = [
    case
    for case in GOLDEN_CASES
    if "ToolCorrectness" in case.get("evaluation_metrics", [])
]


@pytest.mark.llm
@pytest.mark.parametrize("case", REASONING_CASES, ids=lambda case: case["id"])
def test_agent_tool_reasoning_geval(
    case, agent, eval_judge, record_metric
) -> None:
    """The final answer must accurately translate raw tool results."""
    response = agent.run(case["query"])
    tool_summary = json.dumps(
        [tool.model_dump() for tool in response.tools_called], ensure_ascii=False
    )

    test_case = LLMTestCase(
        input=f"User Query: {case['query']}\nTools Executed: {tool_summary}",
        actual_output=response.output,
        expected_output=case["ground_truth"],
    )

    metric = GEval(
        name="ToolReasoningAndResolution",
        criteria=(
            "Judge whether the response converts the raw tool data into a correct, "
            "actionable, and polite answer. The response passes when it (1) states the "
            "same decisive facts as the expected output, including order status, dates, "
            "amounts, fees, and the eligibility decision; (2) does not contradict the "
            "tool data or invent amounts, dates, or policies; and (3) directly resolves "
            "the user's request. Extra correct details (for example the item name) and "
            "different wording or tone are acceptable and must not lower the score. "
            "Penalize only factual errors, missing decisive facts, contradictions with "
            "the expected output, or an unhelpful or unsafe answer."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=eval_judge,
        threshold=0.85,
        async_mode=False,
    )
    measure_and_record(
        metric,
        test_case,
        record_metric,
        f"{case['id']}_tool_reasoning_geval",
        metric_name="ToolReasoningGEval",
    )
