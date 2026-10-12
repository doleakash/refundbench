from types import SimpleNamespace

from benchmark import runner
from benchmark.metrics import (
	calculate_business_impact_metrics,
	calculate_decision_metrics,
	calculate_metrics,
	calculate_performance_metrics,
)
from benchmark.models import (
	BenchmarkResult,
	CasePerformance,
	FieldComparison,
)
from app.infrastructure.observability import TimingEvent, log_timing_event


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


def test_metrics_empty_result_set_marks_rates_unavailable():
	metrics = calculate_metrics([])

	assert metrics.pass_rate is None
	assert metrics.intent_accuracy is None
	assert metrics.grievance_accuracy is None
	assert metrics.judgement_accuracy is None
	assert metrics.refund_decision_accuracy is None
	assert metrics.refund_amount_accuracy is None
	assert metrics.escalation_accuracy is None
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


def test_metrics_do_not_count_execution_failures_as_passes():
	result = make_result("CASE-FAILED-EXECUTION", True, [])
	result.execution_status = "failed"

	metrics = calculate_metrics([result])

	assert metrics.pass_rate == 0
	assert metrics.execution_failure_count == 1
	assert metrics.completed_case_count == 0


def test_business_metrics_track_workflow_outcomes_separately_from_decisions():
	resolved = BenchmarkResult(
		case_id="CASE-RESOLVED",
		passed=True,
		comparisons=[
			FieldComparison(
				field="outcome",
				expected=None,
				actual="RESOLVED",
				passed=True,
			),
		],
	)
	missing = BenchmarkResult(
		case_id="CASE-OUTCOME-MISSING",
		passed=False,
		comparisons=[],
	)

	metrics = calculate_business_impact_metrics([resolved, missing])

	assert metrics.workflow_outcome_classified_case_count == 1
	assert metrics.workflow_outcome_unavailable_count == 1
	assert metrics.workflow_outcome_distribution == {"RESOLVED": 1}
	assert metrics.estimated_human_review_minutes is None
	assert metrics.estimated_manual_handling_cost_avoided is None
	assert any(
		"no recorded workflow outcome" in reason
		for reason in metrics.savings_unavailable_reasons
	)


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
				llm_retry_count=0,
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
				llm_retry_count=0,
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
				llm_retry_count=0,
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


def test_performance_metrics_empty_results_mark_measurements_unavailable():
	metrics = calculate_performance_metrics([])

	assert metrics.case_latencies_ms == {}
	assert metrics.average_latency_ms is None
	assert metrics.p50_latency_ms is None
	assert metrics.p95_latency_ms is None
	assert metrics.average_llm_call_count is None
	assert metrics.average_judge_call_count is None
	assert metrics.total_retry_count is None
	assert metrics.average_prompt_token_count is None
	assert metrics.average_completion_token_count is None
	assert metrics.average_total_token_count is None
	assert metrics.latency_measured_case_count == 0
	assert metrics.latency_missing_case_count == 0
	assert metrics.latency_coverage is None


def test_performance_metrics_reports_partial_latency_coverage():
	results = [
		BenchmarkResult(
			case_id="CASE-MEASURED",
			passed=True,
			comparisons=[],
			performance=CasePerformance(latency_ms=100),
		),
		BenchmarkResult(
			case_id="CASE-MISSING",
			passed=True,
			comparisons=[],
			performance=CasePerformance(),
		),
		BenchmarkResult(
			case_id="CASE-NO-TELEMETRY",
			passed=False,
			comparisons=[],
		),
	]

	metrics = calculate_performance_metrics(results)

	assert metrics.case_count == 3
	assert metrics.latency_measured_case_count == 1
	assert metrics.latency_missing_case_count == 2
	assert metrics.latency_coverage == 1 / 3
	assert metrics.average_latency_ms == 100
	assert metrics.p50_latency_ms == 100
	assert metrics.p95_latency_ms == 100
	assert metrics.case_latencies_ms == {
		"CASE-MEASURED": 100,
		"CASE-MISSING": None,
		"CASE-NO-TELEMETRY": None,
	}


def _decision_result(
	case_id: str,
	expected_decision: str | None,
	expected_amount: float | None,
	expected_escalation: bool | None,
	actual_decision: str | None,
	actual_amount: float | None,
	actual_escalation: bool | None,
) -> BenchmarkResult:
	expected = {
		"settlement.decision": expected_decision,
		"settlement.refund_amount": expected_amount,
		"escalation": expected_escalation,
	}
	actual = {
		"settlement.decision": actual_decision,
		"settlement.refund_amount": actual_amount,
		"escalation": actual_escalation,
	}
	return BenchmarkResult(
		case_id=case_id,
		passed=expected == actual,
		comparisons=[
			FieldComparison(
				field=field,
				expected=expected_value,
				actual=actual[field],
				passed=expected_value == actual[field],
			)
			for field, expected_value in expected.items()
		],
	)


def test_decision_metrics_classify_all_confusion_matrix_outcomes():
	results = [
		_decision_result("TP", "AUTO_APPROVED", 10, False, "AUTO_APPROVED", 10, False),
		_decision_result("TN", "REJECTED", 0, False, "REJECTED", 0, False),
		_decision_result("FP", "REJECTED", 0, False, "AUTO_APPROVED", 10, False),
		_decision_result("FN", "AUTO_APPROVED", 10, False, "REJECTED", 0, False),
	]

	metrics = calculate_decision_metrics(results)

	assert metrics["decision_confusion_matrix"] == {
		"true_positive": 1,
		"true_negative": 1,
		"false_positive": 1,
		"false_negative": 1,
	}
	assert metrics["decision_confusion_matrix_denominator"] == 4
	assert metrics["decision_confusion_matrix_evaluated_case_count"] == 4
	assert metrics["precision"] == 0.5
	assert metrics["recall"] == 0.5
	assert metrics["specificity"] == 0.5
	assert metrics["f1_score"] == 0.5
	assert metrics["false_positive_rate"] == 0.5
	assert metrics["false_negative_rate"] == 0.5


def test_decision_metrics_include_escalations_and_explain_ambiguous_cases():
	results = [
		_decision_result(
			"EXPECTED-ELIGIBLE-ESCALATED",
			"AUTO_APPROVED",
			10,
			False,
			"ESCALATE",
			10,
			True,
		),
		_decision_result(
			"EXPECTED-INELIGIBLE-ESCALATED",
			"REJECTED",
			0,
			False,
			"ESCALATE",
			0,
			True,
		),
		_decision_result(
			"EXPECTED-UNKNOWN",
			None,
			None,
			None,
			"REJECTED",
			0,
			False,
		),
		_decision_result(
			"ACTUAL-UNKNOWN",
			"AUTO_APPROVED",
			10,
			False,
			None,
			None,
			None,
		),
	]

	metrics = calculate_decision_metrics(results)

	assert metrics["decision_confusion_matrix"] == {
		"true_positive": 0,
		"true_negative": 1,
		"false_positive": 0,
		"false_negative": 1,
	}
	assert metrics["agent_escalation_count"] == 2
	assert metrics["agent_escalations_excluded_from_confusion_matrix"] == 0
	assert metrics["decision_confusion_matrix_excluded_case_count"] == 2
	assert metrics[
		"decision_confusion_matrix_excluded_expected_ambiguous_count"
	] == 1
	assert metrics[
		"decision_confusion_matrix_excluded_actual_ambiguous_count"
	] == 1


def test_decision_metrics_use_null_for_undefined_rates():
	results = [
		_decision_result("TN", "REJECTED", 0, False, "REJECTED", 0, False),
	]

	metrics = calculate_decision_metrics(results)

	assert metrics["precision"] is None
	assert metrics["recall"] is None
	assert metrics["specificity"] == 1
	assert metrics["f1_score"] is None
	assert metrics["false_positive_rate"] == 0
	assert metrics["false_negative_rate"] is None


def make_business_result(
	case_id: str,
	actual_decision: str | None,
	actual_escalation: bool,
	prompt_tokens: int | None = 0,
	completion_tokens: int | None = 0,
) -> BenchmarkResult:
	comparisons = [
		FieldComparison(
			field="outcome",
			expected=None,
			actual=(
				"ESCALATED"
				if actual_escalation or actual_decision == "ESCALATE"
				else "RESOLVED"
			),
			passed=True,
		),
		FieldComparison(
			field="escalation",
			expected=actual_escalation,
			actual=actual_escalation,
			passed=True,
		),
	]
	if actual_decision is not None:
		comparisons.insert(
			0,
			FieldComparison(
				field="settlement.decision",
				expected=actual_decision,
				actual=actual_decision,
				passed=True,
			),
		)
	amount = (
		640
		if actual_decision == "AUTO_APPROVED"
		else 0
	)
	comparisons.append(
		FieldComparison(
			field="settlement.refund_amount",
			expected=amount,
			actual=amount,
			passed=True,
		)
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
	assert metrics.successful_automated_resolution_count == 1
	assert metrics.estimated_manual_handling_cost_avoided == 8
	assert metrics.estimated_human_review_cost == 8
	assert metrics.estimated_net_savings == -8


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
	assert metrics.successful_automated_resolution_count == 2
	assert metrics.estimated_manual_handling_cost_avoided == 16
	assert metrics.estimated_llm_cost is None


def test_business_impact_metrics_missing_decisions_are_not_zero_rate():
	results = [
		make_business_result("CASE-001", None, False),
		make_business_result("CASE-002", "UNKNOWN", False),
	]

	metrics = calculate_business_impact_metrics(results)

	assert metrics.automation_rate is None
	assert metrics.escalation_rate is None
	assert metrics.escalation_count == 0
	assert metrics.outcome_unavailable_count == 2
	assert metrics.automation_rate_denominator == 0
	assert metrics.escalation_rate_denominator == 0


def test_business_impact_rates_disclose_classifiable_case_denominator():
	results = [
		make_business_result("CASE-AUTO", "AUTO_APPROVED", False),
		make_business_result("CASE-UNKNOWN", None, False),
	]

	metrics = calculate_business_impact_metrics(results)

	assert metrics.automation_rate == 1
	assert metrics.escalation_rate == 0
	assert metrics.outcome_classified_case_count == 1
	assert metrics.outcome_unavailable_count == 1
	assert metrics.automation_rate_denominator == 1


def test_business_impact_metrics_empty_results_mark_rates_unavailable():
	metrics = calculate_business_impact_metrics(
		[],
		input_cost_per_million_tokens=2,
		output_cost_per_million_tokens=4,
	)

	assert metrics.automation_rate is None
	assert metrics.escalation_rate is None
	assert metrics.escalation_count == 0
	assert metrics.estimated_human_review_minutes is None
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
	assert metrics.estimated_input_cost == 1000 * 2 / 1_000_000
	assert metrics.estimated_output_cost is None
	assert metrics.estimated_llm_cost_per_case is None

	output_only_metrics = calculate_business_impact_metrics(
		[result],
		output_cost_per_million_tokens=4,
	)

	assert output_only_metrics.estimated_input_cost is None
	assert output_only_metrics.estimated_output_cost == 500 * 4 / 1_000_000
	assert output_only_metrics.estimated_llm_cost is None


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


def test_business_impact_does_not_claim_handling_savings_for_wrong_decisions():
	result = make_business_result(
		"CASE-WRONG-AUTO-APPROVAL",
		"AUTO_APPROVED",
		False,
		prompt_tokens=0,
		completion_tokens=0,
	)
	result.passed = False
	next(
		comparison
		for comparison in result.comparisons
		if comparison.field == "settlement.decision"
	).passed = False

	metrics = calculate_business_impact_metrics(
		[result],
		input_cost_per_million_tokens=0,
		output_cost_per_million_tokens=0,
	)

	assert metrics.automated_count == 1
	assert metrics.successful_automated_resolution_count == 0
	assert metrics.estimated_manual_handling_cost_avoided == 0
	assert metrics.estimated_net_savings == 0


def test_business_impact_requires_evaluated_correct_decision_fields():
	result = make_business_result(
		"CASE-MISSING-COMPARISONS",
		"AUTO_APPROVED",
		False,
		prompt_tokens=0,
		completion_tokens=0,
	)
	result.comparisons = []

	metrics = calculate_business_impact_metrics(
		[result],
		input_cost_per_million_tokens=0,
		output_cost_per_million_tokens=0,
	)

	assert metrics.successful_automated_resolution_count == 0
	assert metrics.estimated_manual_handling_cost_avoided is None


def run_fake_case(monkeypatch, timing_events):
	case = SimpleNamespace(
		case_id="CASE-001",
		category="test",
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
				success=True,
				prompt_tokens=None,
				completion_tokens=None,
				total_tokens=None,
			)
		],
	)

	assert result.performance.prompt_token_count == 0
	assert result.performance.completion_token_count == 0
	assert result.performance.total_token_count == 0


def test_run_case_with_missing_event_tokens_preserves_unknown_counts(
	monkeypatch,
):
	result = run_fake_case(
		monkeypatch,
		[
			SimpleNamespace(
				model="test-model",
				stage="LLM_CALL",
				success=True,
				prompt_tokens=None,
				completion_tokens=0,
				total_tokens=0,
			)
		],
	)

	assert result.performance.prompt_token_count is None
	assert result.performance.completion_token_count == 0


def test_run_case_marks_token_aggregate_incomplete_if_one_call_is_missing(
	monkeypatch,
):
	result = run_fake_case(
		monkeypatch,
		[
			SimpleNamespace(
				model="test-model",
				stage="LLM_CALL",
				success=True,
				prompt_tokens=10,
				completion_tokens=5,
				total_tokens=15,
			),
			SimpleNamespace(
				model="test-model",
				stage="LLM_CALL",
				success=True,
				prompt_tokens=None,
				completion_tokens=2,
				total_tokens=None,
			),
		],
	)

	assert result.performance.llm_call_count == 2
	assert result.performance.prompt_token_count is None
	assert result.performance.completion_token_count == 7
	assert result.performance.total_token_count is None


def test_run_case_captures_retries_and_judge_calls_without_double_counting(
	monkeypatch,
):
	events = [
		TimingEvent(
			case_id="CASE-001",
			order_id="ORDER-001",
			stage="LLM_CALL",
			duration_ms=10,
			success=False,
			model="test-model",
			provider="OPENAI",
			endpoint="https://api.openai.com/v1",
		),
		TimingEvent(
			case_id="CASE-001",
			order_id="ORDER-001",
			stage="LLM_RETRY:1",
			duration_ms=1,
			success=False,
			model="test-model",
			provider="OPENAI",
			endpoint="https://api.openai.com/v1",
		),
		TimingEvent(
			case_id="CASE-001",
			order_id="ORDER-001",
			stage="LLM_JUDGE:GRIEVANCE:1",
			duration_ms=12,
			success=True,
			model="test-model",
			provider="OPENAI",
			endpoint="https://api.openai.com/v1",
			prompt_tokens=10,
			completion_tokens=2,
			total_tokens=12,
		),
		TimingEvent(
			case_id="CASE-001",
			order_id="ORDER-001",
			stage="LLM_JUDGE:GRIEVANCE:1",
			duration_ms=10,
			success=False,
			model=None,
		),
		TimingEvent(
			case_id="CASE-001",
			order_id="ORDER-001",
			stage="TOTAL_AGENT",
			duration_ms=100,
			success=True,
		),
	]

	def process_case(**kwargs):
		for event in events:
			log_timing_event(event)
		return SimpleNamespace(
			timing_events=[events[-1]],
			tool_retry_count={},
			response="A response",
		)

	case = SimpleNamespace(
		case_id="CASE-001",
		category="test",
		order_id="ORDER-001",
		customer_message="A test case",
		preconditions=None,
		expected=None,
	)
	service = SimpleNamespace(process_case=process_case)
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

	result = runner._run_case(case, service)

	assert result.performance.llm_call_count == 2
	assert result.performance.failed_llm_call_count == 1
	assert result.performance.llm_retry_count == 1
	assert result.performance.judge_call_count == 2
	assert result.performance.prompt_token_count is None
	assert result.performance.completion_token_count is None


def test_run_case_preserves_failure_latency_and_llm_telemetry(monkeypatch):
	def process_case(**kwargs):
		log_timing_event(
			TimingEvent(
				case_id="CASE-001",
				order_id="ORDER-001",
				stage="LLM_CALL",
				duration_ms=10,
				success=False,
				model="test-model",
				provider="OPENAI",
				prompt_tokens=None,
				completion_tokens=None,
				total_tokens=None,
			)
		)
		log_timing_event(
			TimingEvent(
				case_id="CASE-001",
				order_id="ORDER-001",
				stage="TOTAL_AGENT",
				duration_ms=100,
				success=False,
			)
		)
		raise RuntimeError("agent execution failed")

	case = SimpleNamespace(
		case_id="CASE-001",
		category="test",
		order_id="ORDER-001",
		customer_message="A test case",
		preconditions=None,
		expected=None,
	)
	service = SimpleNamespace(process_case=process_case)
	monkeypatch.setattr(runner, "_apply_preconditions", lambda *args: None)

	result = runner._run_case(case, service)

	assert result.execution_status == "failed"
	assert result.execution_error == "RuntimeError: agent execution failed"
	assert not any(
		comparison.field == "outcome"
		for comparison in result.comparisons
	)
	assert result.performance.latency_ms == 100
	assert result.performance.llm_call_count == 1
	assert result.performance.failed_llm_call_count == 1
	assert result.performance.prompt_token_count is None
