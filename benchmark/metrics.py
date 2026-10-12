from collections import Counter
from math import isfinite

from pydantic import BaseModel

from benchmark.models import BenchmarkResult


FIELD_NAMES = {
	"intent": "intent_accuracy",
	"grievance_types": "grievance_accuracy",
	"judgements": "judgement_accuracy",
	"settlement.decision": "refund_decision_accuracy",
	"settlement.refund_amount": "refund_amount_accuracy",
	"escalation": "escalation_accuracy",
	"settlement.refund_status": "refund_status_accuracy",
	"settlement.idempotency_key": "idempotency_key_accuracy",
}


class BenchmarkMetrics(BaseModel):
	case_count: int
	completed_case_count: int
	execution_failure_count: int
	pass_rate: float | None
	intent_accuracy: float | None
	grievance_accuracy: float | None
	judgement_accuracy: float | None
	refund_decision_accuracy: float | None
	refund_amount_accuracy: float | None
	escalation_accuracy: float | None
	refund_status_accuracy: float | None
	idempotency_key_accuracy: float | None
	field_evaluation_coverage: dict[str, int]


class PerformanceMetrics(BaseModel):
	case_count: int
	case_latencies_ms: dict[str, float | None]
	latency_measured_case_count: int
	latency_missing_case_count: int
	latency_coverage: float | None
	average_latency_ms: float | None
	p50_latency_ms: float | None
	p95_latency_ms: float | None
	instrumented_case_count: int
	uninstrumented_case_count: int
	llm_call_count: int | None
	failed_llm_call_count: int | None
	judge_call_count: int | None
	llm_retry_count: int | None
	tool_retry_count: int | None
	execution_failure_count: int
	average_llm_call_count: float | None
	average_judge_call_count: float | None
	total_retry_count: int | None
	prompt_token_case_count: int
	completion_token_case_count: int
	total_token_case_count: int
	total_prompt_tokens: int | None
	total_completion_tokens: int | None
	total_tokens: int | None
	average_prompt_token_count: float | None
	average_completion_token_count: float | None
	average_total_token_count: float | None


class BusinessImpactMetrics(BaseModel):
	case_count: int
	automated_count: int
	successful_automated_resolution_count: int
	escalation_count: int
	other_outcome_count: int
	outcome_unavailable_count: int
	outcome_classified_case_count: int
	workflow_outcome_classified_case_count: int
	workflow_outcome_unavailable_count: int
	workflow_outcome_distribution: dict[str, int]
	workflow_escalation_count: int
	automation_rate_denominator: int
	escalation_rate_denominator: int
	automation_rate: float | None
	escalation_rate: float | None
	estimated_human_review_minutes: float | None
	estimated_human_review_cost: float | None
	estimated_manual_handling_cost_avoided: float | None
	total_input_tokens: int | None
	total_output_tokens: int | None
	estimated_input_cost: float | None
	estimated_output_cost: float | None
	estimated_llm_cost: float | None
	estimated_llm_cost_per_case: float | None
	estimated_net_savings: float | None
	estimated_savings_per_case: float | None
	savings_unavailable_reasons: list[str]


def _comparison_values(result: BenchmarkResult) -> dict[str, object]:
	return {
		comparison.field: comparison.actual
		for comparison in result.comparisons
	}


def _outcome_kind(result: BenchmarkResult) -> str:
	values = _comparison_values(result)
	decision = values.get("settlement.decision")
	escalation = values.get("escalation")
	amount = values.get("settlement.refund_amount")

	if escalation is True or decision == "ESCALATE":
		return "escalated"
	if decision == "AUTO_APPROVED":
		if (
			isinstance(amount, (int, float))
			and not isinstance(amount, bool)
			and isfinite(amount)
			and amount > 0
			and escalation is False
		):
			return "automated"
		if (
			isinstance(amount, (int, float))
			and not isinstance(amount, bool)
			and amount == 0
			and escalation is False
		):
			return "other"
		return "unavailable"
	if decision in {
		"PROPOSED",
		"PROCESSING",
		"REFUNDED",
		"REJECTED",
		"REJECT",
		"DENIED",
		"DECLINED",
		"NO_REFUND",
		"NOT_APPROVED",
	} and escalation is False:
		return "other"
	return "unavailable"


def _automatic_decision_correct(result: BenchmarkResult) -> bool:
	required_fields = {
		"settlement.decision",
		"settlement.refund_amount",
		"escalation",
	}
	comparisons = {
		comparison.field: comparison
		for comparison in result.comparisons
	}
	return (
		_comparison_values(result).get("outcome") == "RESOLVED"
		and required_fields.issubset(comparisons)
		and all(comparisons[field].passed for field in required_fields)
	)


def _token_totals(
	results: list[BenchmarkResult],
	field: str,
) -> tuple[int | None, int]:
	observed = [
		getattr(result.performance, field)
		for result in results
		if result.performance is not None
	]
	available = [value for value in observed if value is not None]
	if not available or len(available) != len(results):
		return None, len(available)
	return sum(available), len(available)


def calculate_metrics(
	results: list[BenchmarkResult],
) -> BenchmarkMetrics:
	field_comparisons = {
		field: [
			comparison
			for result in results
			for comparison in result.comparisons
			if comparison.field == field
		]
		for field in FIELD_NAMES
	}

	def accuracy(field: str) -> float | None:
		comparisons = field_comparisons[field]
		if not comparisons:
			return None
		return sum(comparison.passed for comparison in comparisons) / len(
			comparisons
		)

	coverage = {
		field: len(comparisons)
		for field, comparisons in field_comparisons.items()
	}
	case_count = len(results)
	completed_count = sum(
		result.execution_status == "completed" for result in results
	)
	return BenchmarkMetrics(
		case_count=case_count,
		completed_case_count=completed_count,
		execution_failure_count=case_count - completed_count,
		pass_rate=(
			sum(
				result.passed and result.execution_status == "completed"
				for result in results
			)
			/ case_count
			if case_count
			else None
		),
		**{
			metric_name: accuracy(field)
			for field, metric_name in FIELD_NAMES.items()
		},
		field_evaluation_coverage=coverage,
	)


def calculate_business_impact_metrics(
	results: list[BenchmarkResult],
	review_minutes_per_escalation: float = 5,
	input_cost_per_million_tokens: float | None = None,
	output_cost_per_million_tokens: float | None = None,
	manual_handling_cost_per_case: float | None = 8,
	human_review_cost_per_hour: float | None = 30,
) -> BusinessImpactMetrics:
	if review_minutes_per_escalation < 0:
		raise ValueError("Review minutes per escalation cannot be negative.")
	for label, value in (
		("Input token price", input_cost_per_million_tokens),
		("Output token price", output_cost_per_million_tokens),
		("Manual handling cost", manual_handling_cost_per_case),
		("Human-review hourly cost", human_review_cost_per_hour),
	):
		if value is not None and value < 0:
			raise ValueError(f"{label} cannot be negative.")

	outcome_kinds = [_outcome_kind(result) for result in results]
	workflow_outcomes = [
		_comparison_values(result).get("outcome")
		for result in results
	]
	workflow_outcome_distribution = dict(
		Counter(value for value in workflow_outcomes if isinstance(value, str))
	)
	workflow_outcome_unavailable_count = sum(
		value is None for value in workflow_outcomes
	)
	automated_count = outcome_kinds.count("automated")
	escalation_count = outcome_kinds.count("escalated")
	other_count = outcome_kinds.count("other")
	unavailable_count = outcome_kinds.count("unavailable")
	workflow_escalation_count = workflow_outcomes.count("ESCALATED")
	correct_automations = sum(
		kind == "automated"
		and _automatic_decision_correct(result)
		and result.execution_status == "completed"
		for kind, result in zip(outcome_kinds, results)
	)
	case_count = len(results)
	outcome_classified_count = case_count - unavailable_count

	total_input_tokens, _ = _token_totals(results, "prompt_token_count")
	total_output_tokens, _ = _token_totals(
		results, "completion_token_count"
	)
	estimated_input_cost = (
		total_input_tokens * input_cost_per_million_tokens / 1_000_000
		if total_input_tokens is not None
		and input_cost_per_million_tokens is not None
		else None
	)
	estimated_output_cost = (
		total_output_tokens * output_cost_per_million_tokens / 1_000_000
		if total_output_tokens is not None
		and output_cost_per_million_tokens is not None
		else None
	)
	estimated_llm_cost = (
		estimated_input_cost + estimated_output_cost
		if estimated_input_cost is not None
		and estimated_output_cost is not None
		else None
	)

	estimated_review_minutes = (
		workflow_escalation_count * review_minutes_per_escalation
		if results and not workflow_outcome_unavailable_count
		else None
	)
	estimated_review_cost = (
		estimated_review_minutes * human_review_cost_per_hour / 60
		if estimated_review_minutes is not None
		and human_review_cost_per_hour is not None
		else None
	)
	handling_cost_avoided = (
		correct_automations * manual_handling_cost_per_case
		if manual_handling_cost_per_case is not None
		and not workflow_outcome_unavailable_count
		else None
	)

	unavailable_reasons = []
	if not results:
		unavailable_reasons.append("No benchmark cases were evaluated.")
	if unavailable_count:
		unavailable_reasons.append(
			f"{unavailable_count} case outcome(s) could not be classified."
		)
	if workflow_outcome_unavailable_count:
		unavailable_reasons.append(
			f"{workflow_outcome_unavailable_count} case(s) have no recorded "
			"workflow outcome; handling savings and review costs are unavailable."
		)
	if manual_handling_cost_per_case is None:
		unavailable_reasons.append(
			"Manual handling cost per case is not configured."
		)
	if human_review_cost_per_hour is None:
		unavailable_reasons.append(
			"Human-review labor cost per hour is not configured."
		)
	if estimated_llm_cost is None:
		unavailable_reasons.append(
			"Complete token telemetry and both token prices are required."
		)
	net_savings = None
	if (
		results
		and not unavailable_count
		and handling_cost_avoided is not None
		and estimated_review_cost is not None
		and estimated_llm_cost is not None
	):
		net_savings = (
			handling_cost_avoided
			- estimated_review_cost
			- estimated_llm_cost
		)

	return BusinessImpactMetrics(
		case_count=case_count,
		automated_count=automated_count,
		successful_automated_resolution_count=correct_automations,
		escalation_count=escalation_count,
		other_outcome_count=other_count,
		outcome_unavailable_count=unavailable_count,
		outcome_classified_case_count=outcome_classified_count,
		workflow_outcome_classified_case_count=(
			case_count - workflow_outcome_unavailable_count
		),
		workflow_outcome_unavailable_count=workflow_outcome_unavailable_count,
		workflow_outcome_distribution=workflow_outcome_distribution,
		workflow_escalation_count=workflow_escalation_count,
		automation_rate=(
			automated_count / outcome_classified_count
			if outcome_classified_count
			else None
		),
		escalation_rate=(
			escalation_count / outcome_classified_count
			if outcome_classified_count
			else None
		),
		automation_rate_denominator=outcome_classified_count,
		escalation_rate_denominator=outcome_classified_count,
		estimated_human_review_minutes=estimated_review_minutes,
		estimated_human_review_cost=estimated_review_cost,
		estimated_manual_handling_cost_avoided=handling_cost_avoided,
		total_input_tokens=total_input_tokens,
		total_output_tokens=total_output_tokens,
		estimated_input_cost=estimated_input_cost,
		estimated_output_cost=estimated_output_cost,
		estimated_llm_cost=estimated_llm_cost,
		estimated_llm_cost_per_case=(
			estimated_llm_cost / case_count
			if estimated_llm_cost is not None and case_count
			else None
		),
		estimated_net_savings=net_savings,
		estimated_savings_per_case=(
			net_savings / case_count if net_savings is not None else None
		),
		savings_unavailable_reasons=unavailable_reasons,
	)


def calculate_performance_metrics(
	results: list[BenchmarkResult],
) -> PerformanceMetrics:
	def valid_latency(value: float | None) -> bool:
		return value is not None and isfinite(value) and value >= 0

	latencies = sorted(
		result.performance.latency_ms
		for result in results
		if result.performance is not None
		and valid_latency(result.performance.latency_ms)
	)

	def percentile(percent: float) -> float | None:
		if not latencies:
			return None
		position = (len(latencies) - 1) * percent
		lower = int(position)
		upper = min(lower + 1, len(latencies) - 1)
		fraction = position - lower
		return (
			latencies[lower] * (1 - fraction)
			+ latencies[upper] * fraction
		)

	total = len(results)
	latency_count = len(latencies)
	instrumented = [
		result.performance
		for result in results
		if result.performance is not None
		and result.performance.llm_call_count is not None
	]

	def observed_sum(field: str) -> int | None:
		values = [
			getattr(item, field)
			for item in instrumented
			if getattr(item, field) is not None
		]
		return (
			sum(values)
			if len(instrumented) == total
			and len(values) == total
			and total
			else None
		)

	def observed_average(field: str) -> float | None:
		values = [
			getattr(item, field)
			for item in instrumented
			if getattr(item, field) is not None
		]
		return (
			sum(values) / len(values)
			if total and len(instrumented) == total and len(values) == total
			else None
		)

	prompt_total, prompt_coverage = _token_totals(
		results, "prompt_token_count"
	)
	completion_total, completion_coverage = _token_totals(
		results, "completion_token_count"
	)
	total_total, total_coverage = _token_totals(results, "total_token_count")
	tool_retries = observed_sum("retry_count")
	llm_retries = observed_sum("llm_retry_count")
	return PerformanceMetrics(
		case_count=total,
		case_latencies_ms={
			result.case_id: (
				result.performance.latency_ms
				if result.performance is not None
				and valid_latency(result.performance.latency_ms)
				else None
			)
			for result in results
		},
		latency_measured_case_count=latency_count,
		latency_missing_case_count=total - latency_count,
		latency_coverage=latency_count / total if total else None,
		average_latency_ms=(
			sum(latencies) / latency_count if latencies else None
		),
		p50_latency_ms=percentile(0.5),
		p95_latency_ms=percentile(0.95),
		instrumented_case_count=len(instrumented),
		uninstrumented_case_count=total - len(instrumented),
		llm_call_count=observed_sum("llm_call_count"),
		failed_llm_call_count=observed_sum("failed_llm_call_count"),
		judge_call_count=observed_sum("judge_call_count"),
		llm_retry_count=observed_sum("llm_retry_count"),
		tool_retry_count=observed_sum("retry_count"),
		execution_failure_count=sum(
			result.execution_status != "completed" for result in results
		),
		average_llm_call_count=observed_average("llm_call_count"),
		average_judge_call_count=observed_average("judge_call_count"),
		total_retry_count=(
			tool_retries + llm_retries
			if tool_retries is not None and llm_retries is not None
			else None
		),
		prompt_token_case_count=prompt_coverage,
		completion_token_case_count=completion_coverage,
		total_token_case_count=total_coverage,
		total_prompt_tokens=prompt_total,
		total_completion_tokens=completion_total,
		total_tokens=total_total,
		average_prompt_token_count=(
			prompt_total / total if prompt_total is not None and total else None
		),
		average_completion_token_count=(
			completion_total / total
			if completion_total is not None and total
			else None
		),
		average_total_token_count=(
			total_total / total if total_total is not None and total else None
		),
	)


def calculate_decision_metrics(
	results: list[BenchmarkResult],
) -> dict[str, object]:
	"""Classify automatic approval as the positive refund-decision class."""
	counts = {
		"true_positive": 0,
		"true_negative": 0,
		"false_positive": 0,
		"false_negative": 0,
	}
	expected_ambiguous = 0
	actual_ambiguous = 0
	excluded_count = 0
	agent_escalations = 0
	for result in results:
		expected_values = {
			comparison.field: comparison.expected
			for comparison in result.comparisons
		}
		actual_values = _comparison_values(result)
		expected = automatic_refund_label(
			expected_values.get("settlement.decision"),
			expected_values.get("settlement.refund_amount"),
			expected_values.get("escalation"),
		)
		actual = automatic_refund_label(
			actual_values.get("settlement.decision"),
			actual_values.get("settlement.refund_amount"),
			actual_values.get("escalation"),
		)
		if actual_values.get("escalation") is True or actual_values.get(
			"settlement.decision"
		) == "ESCALATE":
			agent_escalations += 1
			actual = False
		if expected is None:
			expected_ambiguous += 1
		if actual is None:
			actual_ambiguous += 1
		if expected is None or actual is None:
			excluded_count += 1
			continue
		if expected and actual:
			counts["true_positive"] += 1
		elif not expected and not actual:
			counts["true_negative"] += 1
		elif not expected and actual:
			counts["false_positive"] += 1
		else:
			counts["false_negative"] += 1

	tp = counts["true_positive"]
	tn = counts["true_negative"]
	fp = counts["false_positive"]
	fn = counts["false_negative"]
	denominator = tp + tn + fp + fn
	excluded_expected = expected_ambiguous
	excluded_actual = sum(
		1
		for result in results
		if automatic_refund_label(
			next(
				(
					item.expected
					for item in result.comparisons
					if item.field == "settlement.decision"
				),
				None,
			),
			next(
				(
					item.expected
					for item in result.comparisons
					if item.field == "settlement.refund_amount"
				),
				None,
			),
			next(
				(
					item.expected
					for item in result.comparisons
					if item.field == "escalation"
				),
				None,
			),
		) is not None
		and automatic_refund_label(
			_comparison_values(result).get("settlement.decision"),
			_comparison_values(result).get("settlement.refund_amount"),
			_comparison_values(result).get("escalation"),
		) is None
	)
	precision_denominator = tp + fp
	recall_denominator = tp + fn
	specificity_denominator = tn + fp
	f1_denominator = 2 * tp + fp + fn
	return {
		"positive_class": "automatically approved eligible refund",
		"decision_confusion_matrix": counts,
		"decision_confusion_matrix_denominator": denominator,
		"decision_confusion_matrix_evaluated_case_count": denominator,
		"decision_confusion_matrix_excluded_case_count": excluded_count,
		"decision_confusion_matrix_excluded_expected_ambiguous_count": excluded_expected,
		"decision_confusion_matrix_excluded_actual_ambiguous_count": max(
			0, excluded_count - excluded_expected
		),
		"decision_confusion_matrix_expected_ambiguous_case_count": expected_ambiguous,
		"decision_confusion_matrix_actual_ambiguous_case_count": actual_ambiguous,
		"agent_escalation_count": agent_escalations,
		"agent_escalations_excluded_from_confusion_matrix": 0,
		"agent_escalations_classified_as_not_automatically_approved": agent_escalations,
		"precision": tp / precision_denominator if precision_denominator else None,
		"recall": tp / recall_denominator if recall_denominator else None,
		"specificity": tn / specificity_denominator if specificity_denominator else None,
		"f1_score": (
			2 * tp / f1_denominator if f1_denominator else None
		),
		"false_positive_count": fp,
		"false_positive_rate": (
			fp / specificity_denominator if specificity_denominator else None
		),
		"false_negative_count": fn,
		"false_negative_rate": (
			fn / recall_denominator if recall_denominator else None
		),
	}


def automatic_refund_label(
	decision: object,
	amount: object,
	escalation: object,
) -> bool | None:
	if escalation is True or decision == "ESCALATE":
		return False
	if decision == "AUTO_APPROVED":
		if (
			isinstance(amount, (int, float))
			and not isinstance(amount, bool)
			and isfinite(amount)
			and amount >= 0
			and escalation is False
		):
			return amount > 0
		return None
	if decision in {
		"PROPOSED",
		"PROCESSING",
		"REFUNDED",
		"REJECTED",
		"REJECT",
		"DENIED",
		"DECLINED",
		"NO_REFUND",
		"NOT_APPROVED",
	} and escalation is False:
		return False
	return None
