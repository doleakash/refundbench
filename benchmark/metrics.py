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
			sum(item.prompt_token_count for item in performance) / total
			if total
			else 0.0
		),
		average_completion_token_count=(
			sum(item.completion_token_count for item in performance) / total
			if total
			else 0.0
		),
		average_total_token_count=(
			sum(item.total_token_count for item in performance) / total
			if total
			else 0.0
		),
	)