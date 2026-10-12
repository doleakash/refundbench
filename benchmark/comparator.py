
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
            field="outcome",
            expected=expected.outcome,
            actual=actual.outcome,
            passed=(
                expected.outcome is None
                or expected.outcome == actual.outcome
            ),
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
            field="escalation",
            expected=expected.escalation,
            actual=actual.escalation,
            passed=expected.escalation == actual.escalation,
        ),
    ]

    if expected.settlement is not None:
        actual_settlement = actual.settlement

        comparisons.extend(
            [
                FieldComparison(
                    field="settlement.decision",
                    expected=expected.settlement.decision,
                    actual=(
                        actual_settlement.decision
                        if actual_settlement is not None
                        else None
                    ),
                    passed=(
                        actual_settlement is not None
                        and expected.settlement.decision
                        == actual_settlement.decision
                    ),
                ),
                FieldComparison(
                    field="settlement.refund_amount",
                    expected=expected.settlement.refund_amount,
                    actual=(
                        actual_settlement.refund_amount
                        if actual_settlement is not None
                        else None
                    ),
                    passed=(
                        actual_settlement is not None
                        and expected.settlement.refund_amount
                        == actual_settlement.refund_amount
                    ),
                ),
            ]
        )

        if expected.settlement.refund_status is not None:
            comparisons.append(
                FieldComparison(
                    field="settlement.refund_status",
                    expected=expected.settlement.refund_status,
                    actual=(
                        actual_settlement.refund_status
                        if actual_settlement is not None
                        else None
                    ),
                    passed=(
                        actual_settlement is not None
                        and expected.settlement.refund_status
                        == actual_settlement.refund_status
                    ),
                )
            )

        if expected.settlement.idempotency_key is not None:
            comparisons.append(
                FieldComparison(
                    field="settlement.idempotency_key",
                    expected=expected.settlement.idempotency_key,
                    actual=(
                        actual_settlement.idempotency_key
                        if actual_settlement is not None
                        else None
                    ),
                    passed=(
                        actual_settlement is not None
                        and expected.settlement.idempotency_key
                        == actual_settlement.idempotency_key
                    ),
                )
            )

    # The benchmark's acceptance criteria are workflow outcome and any
    # explicitly expected settlement. Diagnostic fields (such as intent,
    # grievances, and judgements) remain visible without gating the case.
    outcome_comparison = next(
        comparison
        for comparison in comparisons
        if comparison.field == "outcome"
    )

    settlement_comparisons = [
        comparison
        for comparison in comparisons
        if comparison.field.startswith("settlement.")
    ]

    passed = outcome_comparison.passed and all(
        comparison.passed
        for comparison in settlement_comparisons
    )

    return BenchmarkResult(
        case_id=case_id,
        passed=passed,
        comparisons=comparisons,
    )
