from types import SimpleNamespace

import pytest

from benchmark.comparator import compare_outcomes
from benchmark import runner
from benchmark.models import (
	ActualGrievance,
	ActualJudgement,
	ActualOutcome,
	ActualSettlement,
	ExpectedGrievance,
	ExpectedJudgement,
	ExpectedOutcome,
	ExpectedSettlement,
)


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

	assert result.case_id == "CASE-001"
	assert result.passed is False
	assert [item.passed for item in result.comparisons] == [
		True,
		True,
		True,
		True,
		False,
		True,
	]
	assert result.comparisons[4].field == "settlement.refund_amount"
	assert result.comparisons[4].expected == 640
	assert result.comparisons[4].actual == 320


def test_loader_accepts_case_level_golden_set():
	cases = runner.load_golden_cases()

	assert len(cases) == 15
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
