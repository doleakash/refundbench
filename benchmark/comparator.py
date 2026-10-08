from benchmark.models import (
	ActualOutcome,
	BenchmarkResult,
	ExpectedOutcome,
	FieldComparison,
)


def compare_outcomes(
	case_id: str,
	expected: ExpectedOutcome,
	actual: ActualOutcome,
) -> BenchmarkResult:
	expected_grievance_types = [
		grievance.type for grievance in expected.grievances
	]
	actual_grievance_types = [
		grievance.type for grievance in actual.grievances
	]
	expected_judgements = [
		judgement.model_dump()
		for judgement in expected.judgements
	]
	actual_judgements = [
		judgement.model_dump()
		for judgement in actual.judgements
	]
	comparisons = [
		FieldComparison(
			field="intent",
			expected=expected.intent,
			actual=actual.intent,
			passed=expected.intent == actual.intent,
		),
		FieldComparison(
			field="grievance_types",
			expected=expected_grievance_types,
			actual=actual_grievance_types,
			passed=expected_grievance_types == actual_grievance_types,
		),
		FieldComparison(
			field="judgements",
			expected=expected_judgements,
			actual=actual_judgements,
			passed=expected_judgements == actual_judgements,
		),
		FieldComparison(
			field="settlement.decision",
			expected=expected.settlement.decision,
			actual=actual.settlement.decision,
			passed=(
				expected.settlement.decision == actual.settlement.decision
			),
		),
		FieldComparison(
			field="settlement.refund_amount",
			expected=expected.settlement.refund_amount,
			actual=actual.settlement.refund_amount,
			passed=(
				expected.settlement.refund_amount
				== actual.settlement.refund_amount
			),
		),
		FieldComparison(
			field="escalation",
			expected=expected.escalation,
			actual=actual.escalation,
			passed=expected.escalation == actual.escalation,
		),
	]
	if expected.settlement.refund_status is not None:
		comparisons.append(
			FieldComparison(
				field="settlement.refund_status",
				expected=expected.settlement.refund_status,
				actual=actual.settlement.refund_status,
				passed=(
					expected.settlement.refund_status
					== actual.settlement.refund_status
				),
			)
		)
	if expected.settlement.idempotency_key is not None:
		comparisons.append(
			FieldComparison(
				field="settlement.idempotency_key",
				expected=expected.settlement.idempotency_key,
				actual=actual.settlement.idempotency_key,
				passed=(
					expected.settlement.idempotency_key
					== actual.settlement.idempotency_key
				),
			)
		)
	return BenchmarkResult(
		case_id=case_id,
		passed=all(comparison.passed for comparison in comparisons),
		comparisons=comparisons,
	)
