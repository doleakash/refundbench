import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.agent.state import AgentState
from app.application.cases.service import CaseApplicationService
from app.bootstrap import build_container
from app.policy.engine import PolicyAction
from app.settlement.refund_ledger import RefundRecord
from benchmark.comparator import compare_outcomes
from benchmark.models import (
	ActualJudgement,
	ActualGrievance,
	ActualOutcome,
	ActualSettlement,
	BenchmarkResult,
	GoldenCase,
)

GOLDEN_SET_PATH = (
	Path(__file__).resolve().parent.parent
	/ "data"
	/ "golden_set.json"
)


def load_golden_cases(path: Path = GOLDEN_SET_PATH) -> list[GoldenCase]:
	with path.open(encoding="utf-8") as golden_set_file:
		data = json.load(golden_set_file)
	raw_cases = data if isinstance(data, list) else [data]
	return [GoldenCase.model_validate(case) for case in raw_cases]


def _actual_outcome(
	state: AgentState,
	case_service: CaseApplicationService,
) -> ActualOutcome:
	settlement = state.settlement
	refund = case_service.refund_workflow.settlement.ledger.get_refund(
		state.order_id
	)
	return ActualOutcome(
		intent=state.intent.value,
		grievances=[
			ActualGrievance(
				type=grievance.type.value,
			)
			for grievance in state.grievances
		],
		judgements=[
			ActualJudgement(
				grievance_type=grievance.type.value,
				verdict=state.consensus[grievance.grievance_id],
			)
			for grievance in state.grievances
			if grievance.grievance_id in state.consensus
		],
		settlement=ActualSettlement(
			decision=(
				settlement.status.value
				if settlement is not None
				else (
					"ESCALATE"
					if state.actions
					and state.actions[-1] == "ESCALATE"
					else None
				)
			),
			refund_amount=(
				settlement.total_refund
				if settlement is not None
				else 0.0
			),
			refund_status=refund.status if refund is not None else None,
			idempotency_key=(
				refund.idempotency_key if refund is not None else None
			),
		),
		escalation=(
			bool(state.escalation_reason)
			or (
				bool(state.actions)
				and state.actions[-1] == "ESCALATE"
			)
			or any(
				decision.action is PolicyAction.ESCALATE
				for decision in state.policy_decisions.values()
			)
			or (
				settlement is not None
				and settlement.status.value == "ESCALATE"
			)
		),
	)


def _apply_preconditions(
	golden_case: GoldenCase,
	case_service: CaseApplicationService,
) -> None:
	if golden_case.preconditions is None:
		return
	existing_refund = golden_case.preconditions.existing_refund
	if existing_refund is None:
		return

	ledger = case_service.refund_workflow.settlement.ledger
	refund: RefundRecord = ledger.create_refund(
		case_id=golden_case.case_id,
		order_id=golden_case.order_id,
		amount=existing_refund.amount,
	)
	if refund.amount != existing_refund.amount:
		raise ValueError(
			f"Existing refund amount cannot be seeded for order "
			f"{golden_case.order_id}."
		)
	if refund.idempotency_key != existing_refund.idempotency_key:
		raise ValueError(
			f"Existing refund idempotency key cannot be seeded for order "
			f"{golden_case.order_id}."
		)

	if existing_refund.status == "PROPOSED":
		pass
	elif existing_refund.status in {"PROCESSING", "REFUNDED"}:
		ledger.accept_refund(golden_case.order_id)
		if existing_refund.status == "REFUNDED":
			ledger.mark_refunded(golden_case.order_id)
	else:
		raise ValueError(
			f"Existing refund status cannot be seeded using the refund "
			f"lifecycle API: {existing_refund.status}"
		)

	if refund.status != existing_refund.status:
		raise ValueError(
			f"Existing refund status {existing_refund.status} cannot be "
			f"seeded for order {golden_case.order_id}."
		)
	return


def _run_case(
	golden_case: GoldenCase,
	service: CaseApplicationService,
) -> BenchmarkResult:
	_apply_preconditions(golden_case, service)
	state = service.process_case(
		order_id=golden_case.order_id,
		customer_message=golden_case.customer_message,
	)
	return compare_outcomes(
		case_id=golden_case.case_id,
		expected=golden_case.expected,
		actual=_actual_outcome(state, service),
	)


def run_benchmark(
	golden_set_path: Path = GOLDEN_SET_PATH,
	case_service: CaseApplicationService | None = None,
	case_id: str | None = None,
) -> list[BenchmarkResult]:
	golden_cases = load_golden_cases(golden_set_path)
	if case_id is not None:
		golden_cases = [
			case for case in golden_cases if case.case_id == case_id
		]
		if not golden_cases:
			raise ValueError(f"Golden case not found: {case_id}")

	results = []
	for golden_case in golden_cases:
		if case_service is not None:
			results.append(_run_case(golden_case, case_service))
		else:
			with TemporaryDirectory(prefix="refundbench-") as ledger_dir:
				service = build_container(
					ledger_data_dir=Path(ledger_dir),
				).case_service
				results.append(
					_run_case(golden_case, service)
				)
	return results


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--case",
		help="Run only the golden case with this case ID.",
	)
	args = parser.parse_args()

	try:
		results = run_benchmark(case_id=args.case)
	except ValueError as error:
		parser.error(str(error))

	for result in results:
		print(
			f"{result.case_id}  "
			f"{'PASS' if result.passed else 'FAIL'}"
		)
		if not result.passed:
			print(result)
	print(f"Cases: {len(results)}")
	print(f"Passed: {sum(result.passed for result in results)}")
	print(f"Failed: {sum(not result.passed for result in results)}")


if __name__ == "__main__":
	main()
