from pydantic import BaseModel

from benchmark.models import BenchmarkResult


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