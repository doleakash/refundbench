import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.comparator import compare_outcomes
from benchmark import runner
from benchmark.models import (
    ActualGrievance,
    ActualJudgement,
    ActualOutcome,
    ActualSettlement,
    BenchmarkResult,
    BenchmarkResult,
    CasePerformance,
    ExpectedGrievance,
    ExpectedJudgement,
    ExpectedOutcome,
    ExpectedSettlement,
    FieldComparison,
)
from app.domain.models import CustomerIntent


def test_comparator_reports_field_level_pass_and_fail():
    expected = ExpectedOutcome(
        intent="REFUND_REQUEST",
        grievances=[
            ExpectedGrievance(
                type="LATE_DELIVERY",
                claim="The order was late",
            )
        ],
        judgements=[
            ExpectedJudgement(
                grievance_type="LATE_DELIVERY",
                verdict="UPHELD",
            )
        ],
        settlement=ExpectedSettlement(
            decision="AUTO_APPROVED",
            refund_amount=640,
        ),
        escalation=False,
        answer="A refund was approved.",
    )
    actual = ActualOutcome(
        intent="REFUND_REQUEST",
        grievances=[ActualGrievance(type="LATE_DELIVERY")],
        judgements=[
            ActualJudgement(
                grievance_type="LATE_DELIVERY",
                verdict="UPHELD",
            )
        ],
        settlement=ActualSettlement(
            decision="AUTO_APPROVED",
            refund_amount=320,
        ),
        escalation=False,
    )

    result = compare_outcomes("CASE-001", expected, actual)

    assert [item.passed for item in result.comparisons] == [
        True,  # intent
        True,  # outcome
        True,  # grievance_types
        True,  # judgements
        True,  # escalation
        True,  # settlement.decision
        False,  # settlement.refund_amount
    ]

    assert result.comparisons[6].field == "settlement.refund_amount"
    assert result.comparisons[6].expected == 640
    assert result.comparisons[6].actual == 320


def test_deflection_passes_on_outcome_and_keeps_intent_mismatch_diagnostic():
    expected = ExpectedOutcome(
        intent="REFUND_REQUEST",
        grievances=[],
        judgements=[],
        settlement=None,
        escalation=False,
        answer="The cancelled order is outside this workflow.",
        outcome="DEFLECTED",
    )
    actual = ActualOutcome(
        intent="GENERAL_SUPPORT",
        grievances=[],
        judgements=[],
        settlement=None,
        escalation=False,
        outcome="DEFLECTED",
    )

    result = compare_outcomes("CASE-GOLDEN-016", expected, actual)

    assert result.passed is True
    comparisons = {item.field: item for item in result.comparisons}
    assert comparisons["intent"].passed is False
    assert comparisons["intent"].expected == "REFUND_REQUEST"
    assert comparisons["intent"].actual == "GENERAL_SUPPORT"
    assert comparisons["outcome"].passed is True
    assert not any(
        item.field.startswith("settlement.")
        for item in result.comparisons
    )


def test_loader_accepts_case_level_golden_set():
    cases = runner.load_golden_cases()

    assert len(cases) == 16
    assert cases[0].expected.intent == "REFUND_REQUEST"


def test_run_benchmark_selects_only_requested_case(monkeypatch):
    cases = runner.load_golden_cases()
    selected = cases[9]
    processed = []

    class FakeService:
        def process_case(self, order_id, customer_message):
            processed.append((order_id, customer_message))
            return SimpleNamespace(
                timing_events=[],
                tool_retry_count={},
            )

    monkeypatch.setattr(
        runner,
        "_actual_outcome",
        lambda state, service: "actual",
    )
    monkeypatch.setattr(
        runner,
        "compare_outcomes",
        lambda case_id, expected, actual: SimpleNamespace(
            case_id=case_id,
            passed=True,
        ),
    )

    results = runner.run_benchmark(
        case_service=FakeService(),
        case_id="CASE-GOLDEN-010",
    )

    assert [result.case_id for result in results] == [selected.case_id]
    assert processed == [
        (selected.order_id, selected.customer_message)
    ]


def test_run_benchmark_reports_unknown_case():
    with pytest.raises(
        ValueError,
        match="Golden case not found: CASE-NOT-FOUND",
    ):
        runner.run_benchmark(case_id="CASE-NOT-FOUND")


def test_runner_preserves_state_workflow_outcome_in_actual_result():
    state = SimpleNamespace(
        case_id="CASE-RESOLVED",
        order_id="ORD-1",
        intent=CustomerIntent.REFUND_REQUEST,
        outcome="RESOLVED",
        settlement=None,
        actions=["STOP"],
        escalation_reason=None,
        policy_decisions={},
        grievances=[],
        consensus={},
    )
    service = SimpleNamespace(
        refund_workflow=SimpleNamespace(
            settlement=SimpleNamespace(
                ledger=SimpleNamespace(get_refund=lambda order_id: None),
            ),
        ),
    )

    actual = runner._actual_outcome(state, service)

    assert actual.outcome == "RESOLVED"


def test_dashboard_artifact_relative_path_uses_project_root():
    assert runner.resolve_project_path(
        Path("data/benchmark_results")
    ) == runner.PROJECT_ROOT / "data" / "benchmark_results"


def test_cli_displays_business_impact_and_configured_cost(
    monkeypatch, capsys, tmp_path
):
    result = BenchmarkResult(
        case_id="CASE-001",
        passed=True,
        comparisons=[
            FieldComparison(
                field="settlement.decision",
                expected="AUTO_APPROVED",
                actual="AUTO_APPROVED",
                passed=True,
            ),
            FieldComparison(
                field="escalation",
                expected=False,
                actual=False,
                passed=True,
            ),
        ],
        performance=CasePerformance(
            prompt_token_count=1_000_000,
            completion_token_count=500_000,
        ),
    )
    monkeypatch.setattr(runner, "run_benchmark", lambda case_id=None: [result])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.runner",
            "--review-minutes-per-escalation",
            "7",
            "--input-cost-per-million-tokens",
            "2",
            "--output-cost-per-million-tokens",
            "4",
            "--cost-currency",
            "CAD",
            "--dashboard-artifacts-dir",
            str(tmp_path),
        ],
    )

    runner.main()

    output = capsys.readouterr().out
    assert "Business Impact" in output
    assert "Review-time assumption: 7.00 minutes/escalation" in output
    assert "Input price assumption: CAD 2.000000 per million tokens" in output
    assert "Output price assumption: CAD 4.000000 per million tokens" in output
    assert "Estimated LLM cost:    CAD 4.000000" in output
    assert "Estimated cost per case: CAD 4.000000" in output


@pytest.mark.parametrize(
    ("model", "expected_prices", "expected_cost"),
    [
        ("gpt-5.6-luna", "USD 0.200000", "USD 0.800000"),
        (
            "gpt-5-luna",
            "N/A (not configured)",
            "N/A (requires token data and both token prices)",
        ),
    ],
)
def test_cli_uses_openai_pricing_defaults_only_for_exact_model(
    monkeypatch,
    capsys,
    tmp_path,
    model,
    expected_prices,
    expected_cost,
):
    result = BenchmarkResult(
        case_id="CASE-001",
        passed=True,
        comparisons=[
            FieldComparison(
                field="settlement.decision",
                expected="AUTO_APPROVED",
                actual="AUTO_APPROVED",
                passed=True,
            ),
            FieldComparison(
                field="escalation",
                expected=False,
                actual=False,
                passed=True,
            ),
        ],
        performance=CasePerformance(
            prompt_token_count=1_000_000,
            completion_token_count=500_000,
        ),
    )
    monkeypatch.setattr(
        runner,
        "get_settings",
        lambda: SimpleNamespace(
            llm_provider="OPENAI",
            open_ai_model=model,
            llm_base_url=None,
        ),
    )
    monkeypatch.setattr(runner, "run_benchmark", lambda case_id=None: [result])
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark.runner", "--dashboard-artifacts-dir", str(tmp_path)],
    )
    for variable in (
        "REFUNDBENCH_INPUT_COST_PER_MILLION_TOKENS",
        "REFUNDBENCH_OUTPUT_COST_PER_MILLION_TOKENS",
        "REFUNDBENCH_COST_CURRENCY",
    ):
        monkeypatch.delenv(variable, raising=False)

    runner.main()

    output = capsys.readouterr().out
    assert f"Input price assumption: {expected_prices}" in output
    if model == "gpt-5.6-luna":
        assert "Output price assumption: USD 1.200000" in output
    assert f"Estimated LLM cost:    {expected_cost}" in output


@pytest.mark.parametrize(
    ("performance", "pricing_args"),
    [
        (
            CasePerformance(
                prompt_token_count=1_000_000,
                completion_token_count=500_000,
            ),
            ["--input-cost-per-million-tokens", "2"],
        ),
        (
            CasePerformance(
                prompt_token_count=None,
                completion_token_count=500_000,
            ),
            [
                "--input-cost-per-million-tokens",
                "2",
                "--output-cost-per-million-tokens",
                "4",
            ],
        ),
    ],
)
def test_cli_displays_unavailable_cost_as_na(
    monkeypatch,
    capsys,
    tmp_path,
    performance,
    pricing_args,
):
    result = BenchmarkResult(
        case_id="CASE-001",
        passed=True,
        comparisons=[
            FieldComparison(
                field="settlement.decision",
                expected="AUTO_APPROVED",
                actual="AUTO_APPROVED",
                passed=True,
            ),
            FieldComparison(
                field="escalation",
                expected=False,
                actual=False,
                passed=True,
            ),
        ],
        performance=performance,
    )
    monkeypatch.setattr(
        runner,
        "get_settings",
        lambda: SimpleNamespace(
            llm_provider="OPENAI",
            open_ai_model="unpriced-test-model",
            llm_base_url=None,
        ),
    )
    monkeypatch.setattr(runner, "run_benchmark", lambda case_id=None: [result])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.runner",
            *pricing_args,
            "--dashboard-artifacts-dir",
            str(tmp_path),
        ],
    )

    runner.main()

    output = capsys.readouterr().out
    assert "Business Impact" in output
    assert "Estimated LLM cost:" in output
    assert "N/A (requires token data and both token prices)" in output
    assert "Estimated LLM cost:    USD 0.000000" not in output
