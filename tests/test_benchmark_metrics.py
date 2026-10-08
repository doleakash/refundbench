from benchmark.metrics import (
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


def test_performance_metrics_empty_results_are_zero():
	metrics = calculate_performance_metrics([])

	assert metrics.case_latencies_ms == {}
	assert metrics.average_latency_ms == 0
	assert metrics.p50_latency_ms == 0
	assert metrics.p95_latency_ms == 0
	assert metrics.average_llm_call_count == 0
	assert metrics.average_judge_call_count == 0
	assert metrics.total_retry_count == 0