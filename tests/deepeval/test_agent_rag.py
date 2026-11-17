"""DeepEval RAG metrics: faithfulness, answer relevancy and hallucination."""
from __future__ import annotations

import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

from tests.deepeval._metrics import measure_and_record

pytestmark = pytest.mark.llm


def test_agent_rag_faithfulness(
    golden_cases, agent, eval_judge, record_metric
) -> None:
    """Agent answers must be supported by retrieved context and tool results."""
    for case in golden_cases:
        response = agent.run(case["query"])
        test_case = LLMTestCase(
            input=case["query"],
            actual_output=response.output,
            expected_output=case["ground_truth"],
            retrieval_context=response.retrieval_context or case["expected_context"],
            context=case["expected_context"],
        )
        metric = FaithfulnessMetric(
            threshold=0.85, model=eval_judge, include_reason=True, async_mode=False
        )
        measure_and_record(
            metric,
            test_case,
            record_metric,
            f"{case['id']}_faithfulness",
        )


def test_agent_rag_relevancy(golden_cases, agent, eval_judge, record_metric) -> None:
    """Agent answers must directly address the user's question."""
    for case in golden_cases:
        response = agent.run(case["query"])
        test_case = LLMTestCase(
            input=case["query"],
            actual_output=response.output,
            expected_output=case["ground_truth"],
        )
        metric = AnswerRelevancyMetric(
            threshold=0.80, model=eval_judge, include_reason=True, async_mode=False
        )
        measure_and_record(
            metric,
            test_case,
            record_metric,
            f"{case['id']}_relevancy",
        )


def test_agent_hallucination(golden_cases, agent, eval_judge, record_metric) -> None:
    """Agent answers must not contradict the verified knowledge base."""
    cases = [
        case
        for case in golden_cases
        if "Hallucination" in case.get("evaluation_metrics", [])
    ]
    for case in cases:
        response = agent.run(case["query"])
        test_case = LLMTestCase(
            input=case["query"],
            actual_output=response.output,
            context=case["expected_context"] + response.retrieval_context,
        )
        metric = HallucinationMetric(
            threshold=0.85, model=eval_judge, include_reason=True, async_mode=False
        )
        measure_and_record(
            metric,
            test_case,
            record_metric,
            f"{case['id']}_hallucination",
        )
