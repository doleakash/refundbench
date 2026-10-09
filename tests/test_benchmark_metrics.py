from types import SimpleNamespace

from benchmark import runner
from benchmark.metrics import (
	calculate_business_impact_metrics,
	calculate_metrics,
	calculate_performance_metrics,
)
from benchmark.models import (
	BenchmarkResult,
	CasePerformance,
	FieldComparison,
)


def make_result(
	case_id: str,
	passed: bool,
	comparisons: list[tuple[str, bool]],
) -> BenchmarkResult:
	return BenchmarkResult(
		case_id=case_id,
		passed=passed,
		comparisons=[
			FieldComparison(
				field=field,
				expected=None,
				actual=None,
				passed=field_passed,
			)
			for field, field_passed in comparisons
		],
	)


def test_metrics_calculate_accuracy():
	results = [
		make_result(
			"CASE-001",
			True,
			[
				("intent", True),
				("grievance_types", True),
				("judgements", True),
				("settlement.decision", True),
				("settlement.refund_amount", True),
				("escalation", True),
				("settlement.refund_status", True),
				("settlement.idempotency_key", True),
			],
		),
		make_result(
			"CASE-002",
			False,
			[
				("intent", False),
				("grievance_types", True),
				("judgements", False),
				("settlement.decision", True),
				("settlement.refund_amount", False),
				("escalation", True),
			],
		),
	]

	metrics = calculate_metrics(results)

	assert metrics.pass_rate == 0.5
	assert metrics.intent_accuracy == 0.5
	assert metrics.grievance_accuracy == 1.0
	assert metrics.judgement_accuracy == 0.5
	assert metrics.refund_decision_accuracy == 1.0
	assert metrics.refund_amount_accuracy == 0.5
	assert metrics.escalation_accuracy == 1.0
	assert metrics.refund_status_accuracy == 1.0
	assert metrics.idempotency_key_accuracy == 1.0


def test_metrics_empty_result_set_returns_zero_values():
	metrics = calculate_metrics([])

	assert metrics.pass_rate == 0.0
	assert metrics.intent_accuracy == 0.0
	assert metrics.grievance_accuracy == 0.0
	assert metrics.judgement_accuracy == 0.0
	assert metrics.refund_decision_accuracy == 0.0
	assert metrics.refund_amount_accuracy == 0.0
	assert metrics.escalation_accuracy == 0.0
	assert metrics.refund_status_accuracy is None
	assert metrics.idempotency_key_accuracy is None


def test_metrics_reports_optional_fields_as_not_applicable():
	result = make_result(
		"CASE-003",
		True,
		[
			("intent", True),
			("grievance_types", True),
			("judgements", True),
			("settlement.decision", True),
			("settlement.refund_amount", True),
			("escalation", True),
		],
	)

	metrics = calculate_metrics([result])

	assert metrics.refund_status_accuracy is None
	assert metrics.idempotency_key_accuracy is None


def test_performance_metrics_aggregate_case_execution_data():
	results = [
		BenchmarkResult(
			case_id="CASE-001",
			passed=True,
			comparisons=[],
			performance=CasePerformance(
				latency_ms=100,
				llm_call_count=4,
				judge_call_count=3,
				retry_count=1,
				prompt_token_count=100,
				completion_token_count=40,
				total_token_count=140,
			),
		),
		BenchmarkResult(
			case_id="CASE-002",
			passed=False,
			comparisons=[],
			performance=CasePerformance(
				latency_ms=200,
				llm_call_count=2,
				judge_call_count=1,
				retry_count=2,
				prompt_token_count=80,
				completion_token_count=20,
				total_token_count=100,
			),
		),
		BenchmarkResult(
			case_id="CASE-003",
			passed=True,
			comparisons=[],
			performance=CasePerformance(
				latency_ms=300,
				llm_call_count=0,
				judge_call_count=0,
				retry_count=0,
				prompt_token_count=30,
				completion_token_count=10,
				total_token_count=40,
			),
		),
	]

	metrics = calculate_performance_metrics(results)

	assert metrics.case_latencies_ms == {
		"CASE-001": 100,
		"CASE-002": 200,
		"CASE-003": 300,
	}
	assert metrics.average_latency_ms == 200
	assert metrics.p50_latency_ms == 200
	assert metrics.p95_latency_ms == 290
	assert metrics.average_llm_call_count == 2
	assert metrics.average_judge_call_count == 4 / 3
	assert metrics.total_retry_count == 3
	assert metrics.average_prompt_token_count == 70
	assert metrics.average_completion_token_count == 70 / 3
	assert metrics.average_total_token_count == 280 / 3


def test_performance_metrics_empty_results_are_zero():
	metrics = calculate_performance_metrics([])

	assert metrics.case_latencies_ms == {}
	assert metrics.average_latency_ms == 0
	assert metrics.p50_latency_ms == 0
	assert metrics.p95_latency_ms == 0
	assert metrics.average_llm_call_count == 0
	assert metrics.average_judge_call_count == 0
	assert metrics.total_retry_count == 0
	assert metrics.average_prompt_token_count == 0
	assert metrics.average_completion_token_count == 0
	assert metrics.average_total_token_count == 0


def make_business_result(
	case_id: str,
	actual_decision: str | None,
	actual_escalation: bool,
	prompt_tokens: int | None = 0,
	completion_tokens: int | None = 0,
) -> BenchmarkResult:
	comparisons = [
		FieldComparison(
			field="escalation",
			expected=True,
			actual=actual_escalation,
			passed=False,
		),
	]
	if actual_decision is not None:
		comparisons.insert(
			0,
			FieldComparison(
				field="settlement.decision",
				expected="ESCALATE",
				actual=actual_decision,
				passed=False,
			),
		)
	return BenchmarkResult(
		case_id=case_id,
		passed=True,
		comparisons=comparisons,
		performance=CasePerformance(
			prompt_token_count=prompt_tokens,
			completion_token_count=completion_tokens,
		),
	)


def test_business_impact_metrics_use_actual_outcomes_and_configured_prices():
	results = [
		make_business_result(
			"CASE-001",
			"ESCALATE",
			False,
			prompt_tokens=1_000_000,
			completion_tokens=500_000,
		),
		make_business_result(
			"CASE-002",
			"AUTO_APPROVED",
			True,
			prompt_tokens=500_000,
			completion_tokens=250_000,
		),
		make_business_result(
			"CASE-003",
			"AUTO_APPROVED",
			False,
			prompt_tokens=500_000,
			completion_tokens=250_000,
		),
	]

	metrics = calculate_business_impact_metrics(
		results,
		review_minutes_per_escalation=8,
		input_cost_per_million_tokens=2,
		output_cost_per_million_tokens=4,
	)

	assert metrics.automation_rate == 1 / 3
	assert metrics.escalation_rate == 2 / 3
	assert metrics.escalation_count == 2
	assert metrics.estimated_human_review_minutes == 16
	assert metrics.total_input_tokens == 2_000_000
	assert metrics.total_output_tokens == 1_000_000
	assert metrics.estimated_input_cost == 4
	assert metrics.estimated_output_cost == 4
	assert metrics.estimated_llm_cost == 8
	assert metrics.estimated_llm_cost_per_case == 8 / 3


def test_business_impact_metrics_with_zero_escalations():
	results = [
		make_business_result("CASE-001", "AUTO_APPROVED", False),
		make_business_result("CASE-002", "AUTO_APPROVED", False),
	]

	metrics = calculate_business_impact_metrics(results)

	assert metrics.automation_rate == 1
	assert metrics.escalation_rate == 0
	assert metrics.escalation_count == 0
	assert metrics.estimated_human_review_minutes == 0
	assert metrics.estimated_llm_cost is None


def test_business_impact_metrics_missing_or_invalid_decision_is_not_automated():
	results = [
		make_business_result("CASE-001", None, False),
		make_business_result("CASE-002", "UNKNOWN", False),
	]

	metrics = calculate_business_impact_metrics(results)

	assert metrics.automation_rate == 0
	assert metrics.escalation_rate == 0
	assert metrics.escalation_count == 0


def test_business_impact_metrics_empty_results_are_zero():
	metrics = calculate_business_impact_metrics(
		[],
		input_cost_per_million_tokens=2,
		output_cost_per_million_tokens=4,
	)

	assert metrics.automation_rate == 0
	assert metrics.escalation_rate == 0
	assert metrics.escalation_count == 0
	assert metrics.estimated_human_review_minutes == 0
	assert metrics.total_input_tokens is None
	assert metrics.total_output_tokens is None
	assert metrics.estimated_input_cost is None
	assert metrics.estimated_output_cost is None
	assert metrics.estimated_llm_cost is None
	assert metrics.estimated_llm_cost_per_case is None


def test_business_impact_metrics_cost_is_unavailable_without_both_prices():
	result = make_business_result(
		"CASE-001",
		"AUTO_APPROVED",
		False,
		prompt_tokens=1000,
		completion_tokens=500,
	)

	metrics = calculate_business_impact_metrics(
		[result],
		input_cost_per_million_tokens=2,
	)

	assert metrics.estimated_llm_cost is None
	assert metrics.estimated_input_cost is None
	assert metrics.estimated_output_cost is None
	assert metrics.estimated_llm_cost_per_case is None


def test_business_impact_metrics_cost_is_unavailable_without_token_data():
	for result in (
		make_business_result(
			"CASE-PROMPT-MISSING",
			"AUTO_APPROVED",
			False,
			prompt_tokens=None,
			completion_tokens=500,
		),
		make_business_result(
			"CASE-COMPLETION-MISSING",
			"AUTO_APPROVED",
			False,
			prompt_tokens=1000,
			completion_tokens=None,
		),
	):
		metrics = calculate_business_impact_metrics(
			[result],
			input_cost_per_million_tokens=2,
			output_cost_per_million_tokens=4,
		)

		assert metrics.estimated_llm_cost is None


def test_business_impact_metrics_zero_token_usage_has_zero_cost():
	result = make_business_result(
		"CASE-ZERO-TOKENS",
		"AUTO_APPROVED",
		False,
		prompt_tokens=0,
		completion_tokens=0,
	)

	metrics = calculate_business_impact_metrics(
		[result],
		input_cost_per_million_tokens=2,
		output_cost_per_million_tokens=4,
	)

	assert metrics.estimated_llm_cost == 0
	assert metrics.estimated_input_cost == 0
	assert metrics.estimated_output_cost == 0
	assert metrics.estimated_llm_cost_per_case == 0


def run_fake_case(monkeypatch, timing_events):
	case = SimpleNamespace(
		case_id="CASE-001",
		order_id="ORDER-001",
		customer_message="A test case",
		preconditions=None,
		expected=None,
	)
	service = SimpleNamespace(
		process_case=lambda **kwargs: SimpleNamespace(
			timing_events=timing_events,
			tool_retry_count={},
		)
	)
	monkeypatch.setattr(runner, "_actual_outcome", lambda state, service: None)
	monkeypatch.setattr(
		runner,
		"compare_outcomes",
		lambda **kwargs: BenchmarkResult(
			case_id="CASE-001",
			passed=True,
			comparisons=[],
		),
	)

	return runner._run_case(case, service)


def test_run_case_without_llm_events_preserves_unknown_token_usage(monkeypatch):
	result = run_fake_case(monkeypatch, [])

	assert result.performance.prompt_token_count is None
	assert result.performance.completion_token_count is None


def test_run_case_with_instrumented_no_llm_events_records_zero_tokens(
	monkeypatch,
):
	result = run_fake_case(
		monkeypatch,
		[
			SimpleNamespace(
				model=None,
				stage="GET_ORDER",
				prompt_tokens=None,
				completion_tokens=None,
				total_tokens=None,
			)
		],
	)

	assert result.performance.prompt_token_count == 0
	assert result.performance.completion_token_count == 0


def test_run_case_with_missing_event_tokens_preserves_unknown_counts(
	monkeypatch,
):
	result = run_fake_case(
		monkeypatch,
		[
			SimpleNamespace(
				model="test-model",
				stage="LLM_CALL",
				prompt_tokens=None,
				completion_tokens=0,
				total_tokens=0,
			)
		],
	)

	assert result.performance.prompt_token_count is None
	assert result.performance.completion_token_count == 0