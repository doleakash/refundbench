from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from pydantic import BaseModel

from app.agent.state import AgentState
from app.application.cases.service import CaseApplicationService
from app.bootstrap import build_container
from app.policy.engine import PolicyAction
from app.settlement.refund_ledger import RefundRecord
from app.settlement.settlement import SettlementStatus

GOLDEN_SET_PATH = (
	Path(__file__).resolve().parents[2]
	/ "data"
	/ "golden_set.json"
)


class ExistingRefund(BaseModel):
	status: str
	amount: float
	idempotency_key: str


class Preconditions(BaseModel):
	existing_refund: ExistingRefund | None = None


class GoldenCase(BaseModel):
	case_id: str
	category: str
	order_id: str
	customer_message: str
	expected: dict[str, Any]
	preconditions: Preconditions | None = None


CaseServiceFactory = Callable[[], CaseApplicationService]


def load_golden_cases(path: Path = GOLDEN_SET_PATH) -> list[GoldenCase]:
	with path.open(encoding="utf-8") as golden_file:
		data = json.load(golden_file)
	if not isinstance(data, list):
		raise ValueError("Golden set must contain a list of cases.")
	return [GoldenCase.model_validate(case) for case in data]


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
	refund = ledger.create_refund(
		case_id=golden_case.case_id,
		order_id=golden_case.order_id,
		amount=existing_refund.amount,
	)
	if refund.idempotency_key != existing_refund.idempotency_key:
		raise ValueError(
			f"Precondition idempotency key does not match order "
			f"{golden_case.order_id}."
		)
	if existing_refund.status == "PROCESSING":
		ledger.accept_refund(golden_case.order_id)
	elif existing_refund.status == "REFUNDED":
		ledger.accept_refund(golden_case.order_id)
		ledger.mark_refunded(golden_case.order_id)
	elif existing_refund.status != "PROPOSED":
		raise ValueError(
			f"Unsupported refund precondition status: "
			f"{existing_refund.status}"
		)


def _actual_result(
	state: AgentState,
	case_service: CaseApplicationService,
) -> dict[str, Any]:
	settlement = state.settlement
	refund: RefundRecord | None = (
		case_service.refund_workflow.settlement.ledger.get_refund(
			state.order_id
		)
	)
	escalation = (
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
			and settlement.status == SettlementStatus.ESCALATE
		)
	)
	total_event = next(
		(
			event
			for event in state.timing_events
			if event.stage == "TOTAL_AGENT"
		),
		None,
	)
	llm_events = [
		event
		for event in state.timing_events
		if event.model is not None
	]
	return {
		"intent": state.intent.value,
		"grievances": [
			{"type": grievance.type.value}
			for grievance in state.grievances
		],
		"judgements": [
			{
				"grievance_type": grievance.type.value,
				"verdict": state.consensus[grievance.grievance_id],
			}
			for grievance in state.grievances
			if grievance.grievance_id in state.consensus
		],
		"settlement": {
			"decision": (
				settlement.status.value
				if settlement is not None
				else (
					"ESCALATE"
					if state.actions
					and state.actions[-1] == "ESCALATE"
					else None
				)
			),
			"refund_amount": (
				settlement.total_refund
				if settlement is not None
				else 0.0
			),
			"refund_status": refund.status if refund is not None else None,
			"idempotency_key": (
				refund.idempotency_key if refund is not None else None
			),
		},
		"escalation": escalation,
		"answer": state.response,
		"metrics": {
			"total_latency_ms": (
				total_event.duration_ms if total_event is not None else None
			),
			"llm_call_count": len(llm_events),
			"judge_call_count": sum(
				event.stage.startswith("LLM_JUDGE:")
				for event in llm_events
			),
			"retry_count": sum(state.tool_retry_count.values()),
			"outcome": "escalated" if escalation else "completed",
		},
	}


def compare_case(
	expected: dict[str, Any],
	actual: dict[str, Any],
) -> dict[str, Any]:
	expected_types = Counter(
		grievance["type"] for grievance in expected["grievances"]
	)
	actual_types = Counter(
		grievance["type"] for grievance in actual["grievances"]
	)
	expected_judgements = Counter(
		(item["grievance_type"], item["verdict"])
		for item in expected["judgements"]
	)
	actual_judgements = Counter(
		(item["grievance_type"], item["verdict"])
		for item in actual["judgements"]
	)
	checks = {
		"intent": {
			"passed": expected["intent"] == actual["intent"],
			"expected": expected["intent"],
			"actual": actual["intent"],
		},
		"grievances": {
			"passed": expected_types == actual_types,
			"expected": dict(expected_types),
			"actual": dict(actual_types),
		},
		"judgements": {
			"passed": expected_judgements == actual_judgements,
			"expected": [
				{"grievance_type": kind, "verdict": verdict, "count": count}
				for (kind, verdict), count in sorted(
					expected_judgements.items()
				)
			],
			"actual": [
				{"grievance_type": kind, "verdict": verdict, "count": count}
				for (kind, verdict), count in sorted(
					actual_judgements.items()
				)
			],
		},
		"settlement": {
			"passed": (
				expected["settlement"]["decision"]
				== actual["settlement"]["decision"]
			),
			"expected": expected["settlement"]["decision"],
			"actual": actual["settlement"]["decision"],
		},
		"refund_amount": {
			"passed": (
				expected["settlement"]["refund_amount"]
				== actual["settlement"]["refund_amount"]
			),
			"expected": expected["settlement"]["refund_amount"],
			"actual": actual["settlement"]["refund_amount"],
		},
		"escalation": {
			"passed": expected["escalation"] == actual["escalation"],
			"expected": expected["escalation"],
			"actual": actual["escalation"],
		},
		"answer": {
			"status": "not_scored",
			"expected": expected["answer"],
			"actual": actual["answer"],
		},
	}
	for field in ("refund_status", "idempotency_key"):
		if field in expected["settlement"]:
			expected_value = expected["settlement"][field]
			actual_value = actual["settlement"][field]
			checks[field] = {
				"passed": expected_value == actual_value,
				"expected": expected_value,
				"actual": actual_value,
			}
	return checks


def _run_case(
	golden_case: GoldenCase,
	case_service_factory: CaseServiceFactory,
) -> dict[str, Any]:
	started_at = perf_counter()
	try:
		case_service = case_service_factory()
		_apply_preconditions(golden_case, case_service)
		state = case_service.process_case(
			order_id=golden_case.order_id,
			customer_message=golden_case.customer_message,
		)
		actual = _actual_result(state, case_service)
		checks = compare_case(golden_case.expected, actual)
		return {
			"case_id": golden_case.case_id,
			"category": golden_case.category,
			"status": "PASS" if all(
				check["passed"]
				for check in checks.values()
				if "passed" in check
			) else "FAIL",
			"passed": all(
				check["passed"]
				for check in checks.values()
				if "passed" in check
			),
			"checks": checks,
			"actual_answer": actual["answer"],
			"metrics": actual["metrics"],
		}
	except Exception as error:
		return {
			"case_id": golden_case.case_id,
			"category": golden_case.category,
			"status": "EXECUTION_ERROR",
			"passed": False,
			"error": f"{type(error).__name__}: {error}",
			"checks": {},
			"metrics": {
				"total_latency_ms": (perf_counter() - started_at) * 1000,
				"llm_call_count": 0,
				"judge_call_count": 0,
				"retry_count": 0,
				"outcome": "execution_error",
			},
		}


def run_benchmark(
	golden_set_path: Path = GOLDEN_SET_PATH,
	case_service_factory: CaseServiceFactory | None = None,
) -> list[dict[str, Any]]:
	if case_service_factory is None:
		case_service_factory = lambda: build_container().case_service
	results = []
	for golden_case in load_golden_cases(golden_set_path):
		results.append(_run_case(golden_case, case_service_factory))
	return results


def main() -> None:
	results = run_benchmark()
	for result in results:
		print(f"{result['case_id']}  {result['status']}")
		if result["status"] == "FAIL":
			failed_checks = [
				name
				for name, check in result["checks"].items()
				if check.get("passed") is False
			]
			print(f"  failed checks: {', '.join(failed_checks)}")
		elif result["status"] == "EXECUTION_ERROR":
			print(f"  error: {result['error']}")
	passed = sum(result["status"] == "PASS" for result in results)
	execution_errors = sum(
		result["status"] == "EXECUTION_ERROR" for result in results
	)
	print(f"Cases: {len(results)}")
	print(f"Passed: {passed}")
	print(f"Failed: {len(results) - passed - execution_errors}")
	print(f"Execution errors: {execution_errors}")


if __name__ == "__main__":
	main()
