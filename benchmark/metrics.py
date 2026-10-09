from pydantic import BaseModel

from benchmark.models import BenchmarkResult, CasePerformance


class BenchmarkMetrics(BaseModel):
	pass_rate: float
	intent_accuracy: float
	grievance_accuracy: float
	judgement_accuracy: float
	refund_decision_accuracy: float
	refund_amount_accuracy: float
	escalation_accuracy: float
	refund_status_accuracy: float | None
	idempotency_key_accuracy: float | None


class PerformanceMetrics(BaseModel):
	case_latencies_ms: dict[str, float | None]
	average_latency_ms: float
	p50_latency_ms: float
	p95_latency_ms: float
	average_llm_call_count: float
	average_judge_call_count: float
	total_retry_count: int
	average_prompt_token_count: float
	average_completion_token_count: float
	average_total_token_count: float


class BusinessImpactMetrics(BaseModel):
	automation_rate: float
	escalation_rate: float
	escalation_count: int
	estimated_human_review_minutes: float
	total_input_tokens: int | None
	total_output_tokens: int | None
	estimated_input_cost: float | None
	estimated_output_cost: float | None
	estimated_llm_cost: float | None
	estimated_llm_cost_per_case: float | None


def calculate_metrics(
	results: list[BenchmarkResult],
) -> BenchmarkMetrics:
	total_cases = len(results)

	def accuracy(field: str) -> float:
		comparisons = [
			comparison
			for result in results
			for comparison in result.comparisons
			if comparison.field == field
		]
		if not comparisons:
			return 0.0

		return sum(
			comparison.passed for comparison in comparisons
		) / len(comparisons)

	def optional_accuracy(field: str) -> float | None:
		comparisons = [
			comparison
			for result in results
			for comparison in result.comparisons
			if comparison.field == field
		]
		if not comparisons:
			return None
		return sum(
			comparison.passed for comparison in comparisons
		) / len(comparisons)

	return BenchmarkMetrics(
		pass_rate=(
			sum(result.passed for result in results) / total_cases
			if total_cases
			else 0.0
		),
		intent_accuracy=accuracy("intent"),
		grievance_accuracy=accuracy("grievance_types"),
		judgement_accuracy=accuracy("judgements"),
		refund_decision_accuracy=accuracy("settlement.decision"),
		refund_amount_accuracy=accuracy("settlement.refund_amount"),
		escalation_accuracy=accuracy("escalation"),
		refund_status_accuracy=optional_accuracy(
			"settlement.refund_status"
		),
		idempotency_key_accuracy=optional_accuracy(
			"settlement.idempotency_key"
		),
	)


def calculate_business_impact_metrics(
	results: list[BenchmarkResult],
	review_minutes_per_escalation: float = 5,
	input_cost_per_million_tokens: float | None = None,
	output_cost_per_million_tokens: float | None = None,
) -> BusinessImpactMetrics:
	if review_minutes_per_escalation < 0:
		raise ValueError("Review minutes per escalation cannot be negative.")
	if (
		input_cost_per_million_tokens is not None
		and input_cost_per_million_tokens < 0
	):
		raise ValueError("Input token price cannot be negative.")
	if (
		output_cost_per_million_tokens is not None
		and output_cost_per_million_tokens < 0
	):
		raise ValueError("Output token price cannot be negative.")

	escalation_count = 0
	automation_count = 0
	for result in results:
		actual_values = {
			comparison.field: comparison.actual
			for comparison in result.comparisons
		}
		try:
			actual_escalation = actual_values["escalation"]
		except KeyError as error:
			raise ValueError(
				f"Missing actual {error.args[0]} for case {result.case_id}."
			) from error
		actual_decision = actual_values.get("settlement.decision")

		if actual_decision == "ESCALATE" or actual_escalation is True:
			escalation_count += 1
		if (
			actual_decision == "AUTO_APPROVED"
			and actual_escalation is False
		):
			automation_count += 1

	total_cases = len(results)
	has_token_data = bool(results) and all(
		result.performance is not None
		and result.performance.prompt_token_count is not None
		and result.performance.completion_token_count is not None
		for result in results
	)
	total_input_tokens = None
	total_output_tokens = None
	if has_token_data:
		total_input_tokens = sum(
			result.performance.prompt_token_count
			for result in results
			if result.performance is not None
			and result.performance.prompt_token_count is not None
		)
		total_output_tokens = sum(
			result.performance.completion_token_count
			for result in results
			if result.performance is not None
			and result.performance.completion_token_count is not None
		)

	estimated_input_cost = None
	estimated_output_cost = None
	estimated_llm_cost = None
	if (
		input_cost_per_million_tokens is not None
		and output_cost_per_million_tokens is not None
		and has_token_data
	):
		estimated_input_cost = (
			total_input_tokens * input_cost_per_million_tokens
		) / 1_000_000
		estimated_output_cost = (
			total_output_tokens * output_cost_per_million_tokens
		) / 1_000_000
		estimated_llm_cost = estimated_input_cost + estimated_output_cost

	return BusinessImpactMetrics(
		automation_rate=(
			automation_count / total_cases
			if total_cases
			else 0.0
		),
		escalation_rate=(
			escalation_count / total_cases if total_cases else 0.0
		),
		escalation_count=escalation_count,
		estimated_human_review_minutes=(
			escalation_count * review_minutes_per_escalation
		),
		total_input_tokens=total_input_tokens,
		total_output_tokens=total_output_tokens,
		estimated_input_cost=estimated_input_cost,
		estimated_output_cost=estimated_output_cost,
		estimated_llm_cost=estimated_llm_cost,
		estimated_llm_cost_per_case=(
			estimated_llm_cost / total_cases
			if estimated_llm_cost is not None and total_cases
			else None
		),
	)


def calculate_performance_metrics(
	results: list[BenchmarkResult],
) -> PerformanceMetrics:
	performance = [
		result.performance or CasePerformance()
		for result in results
	]
	latencies = sorted(
		item.latency_ms
		for item in performance
		if item.latency_ms is not None
	)

	def percentile(percent: float) -> float:
		if not latencies:
			return 0.0
		position = (len(latencies) - 1) * percent
		lower = int(position)
		upper = min(lower + 1, len(latencies) - 1)
		fraction = position - lower
		return (
			latencies[lower] * (1 - fraction)
			+ latencies[upper] * fraction
		)

	total = len(results)
	return PerformanceMetrics(
		case_latencies_ms={
			result.case_id: (
				result.performance.latency_ms
				if result.performance is not None
				else None
			)
			for result in results
		},
		average_latency_ms=(
			sum(latencies) / len(latencies) if latencies else 0.0
		),
		p50_latency_ms=percentile(0.5),
		p95_latency_ms=percentile(0.95),
		average_llm_call_count=(
			sum(item.llm_call_count for item in performance) / total
			if total
			else 0.0
		),
		average_judge_call_count=(
			sum(item.judge_call_count for item in performance) / total
			if total
			else 0.0
		),
		total_retry_count=sum(item.retry_count for item in performance),
		average_prompt_token_count=(
			sum(item.prompt_token_count or 0 for item in performance) / total
			if total
			else 0.0
		),
		average_completion_token_count=(
			sum(item.completion_token_count or 0 for item in performance)
			/ total
			if total
			else 0.0
		),
		average_total_token_count=(
			sum(item.total_token_count for item in performance) / total
			if total
			else 0.0
		),
	)