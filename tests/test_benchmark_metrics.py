from benchmark.metrics import calculate_metrics
from benchmark.models import BenchmarkResult, FieldComparison


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