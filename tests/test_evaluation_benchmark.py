import json

import pytest

from app import bootstrap
from app.agent.state import AgentState
from app.domain.models import CustomerIntent
from app.application.cases.repository import InMemoryCaseRepository
from app.application.cases.service import CaseApplicationService
from app.evaluation.benchmark import compare_case, run_benchmark
from app.settlement.settlement import SettlementDecision, SettlementStatus
from app.settlement.refund_workflow import RefundWorkflow


def make_expected():
	return {
		"intent": "REFUND_REQUEST",
		"grievances": [
			{"type": "LATE_DELIVERY", "claim": "Late"},
			{"type": "MISSING_ITEMS", "claim": "Missing"},
		],
		"judgements": [
			{"grievance_type": "LATE_DELIVERY", "verdict": "UPHELD"},
			{"grievance_type": "MISSING_ITEMS", "verdict": "REJECTED"},
		],
		"settlement": {
			"decision": "AUTO_APPROVED",
			"refund_amount": 100,
		},
		"escalation": False,
		"answer": "Expected answer.",
	}


def make_actual():
	return {
		"intent": "REFUND_REQUEST",
		"grievances": [
			{"type": "LATE_DELIVERY"},
			{"type": "MISSING_ITEMS"},
		],
		"judgements": [
			{"grievance_type": "LATE_DELIVERY", "verdict": "UPHELD"},
			{"grievance_type": "MISSING_ITEMS", "verdict": "REJECTED"},
		],
		"settlement": {
			"decision": "AUTO_APPROVED",
			"refund_amount": 100,
			"refund_status": None,
			"idempotency_key": None,
		},
		"escalation": False,
		"answer": "Different but acceptable answer.",
	}


def test_matching_intent_and_unordered_grievances_pass():
	expected = make_expected()
	actual = make_actual()
	actual["grievances"].reverse()

	checks = compare_case(expected, actual)

	assert checks["intent"]["passed"]
	assert checks["grievances"]["passed"]
	assert checks["judgements"]["passed"]
	assert checks["answer"]["status"] == "not_scored"
	assert checks["answer"]["expected"] != checks["answer"]["actual"]


@pytest.mark.parametrize(
	("field", "mutate"),
	[
		("intent", lambda data: data.update(intent="ORDER_SUPPORT")),
		(
			"grievances",
			lambda data: data["grievances"][0].update(
				type="FOOD_QUALITY"
			),
		),
		(
			"judgements",
			lambda data: data["judgements"][0].update(
				verdict="REJECTED"
			),
		),
		(
			"refund_amount",
			lambda data: data["settlement"].update(refund_amount=80),
		),
		("escalation", lambda data: data.update(escalation=True)),
	],
)
def test_mismatches_fail_the_corresponding_check(field, mutate):
	expected = make_expected()
	actual = make_actual()
	mutate(actual)

	checks = compare_case(expected, actual)

	assert not checks[field]["passed"]


def test_precondition_fields_are_compared_when_expected():
	expected = make_expected()
	expected["settlement"].update(
		refund_status="PROCESSING",
		idempotency_key="refund:ORD-123",
	)
	actual = make_actual()
	actual["settlement"].update(
		refund_status="PROCESSING",
		idempotency_key="refund:ORD-123",
	)

	checks = compare_case(expected, actual)

	assert checks["refund_status"]["passed"]
	assert checks["idempotency_key"]["passed"]
	actual["settlement"]["idempotency_key"] = "refund:OTHER"
	assert not compare_case(expected, actual)["idempotency_key"]["passed"]


def test_execution_error_is_recorded_and_later_cases_continue(tmp_path):
	golden_path = tmp_path / "golden.json"
	golden_path.write_text(json.dumps([
		{
			"case_id": "CASE-ERROR",
			"category": "test",
			"order_id": "ORD-123",
			"customer_message": "raise",
			"expected": make_expected(),
		},
		{
			"case_id": "CASE-OK",
			"category": "test",
			"order_id": "ORD-123",
			"customer_message": "complete",
			"expected": {
				"intent": "GENERAL_SUPPORT",
				"grievances": [],
				"judgements": [],
				"settlement": {
					"decision": None,
					"refund_amount": 0,
				},
				"escalation": False,
				"answer": "Expected.",
			},
		},
	]), encoding="utf-8")
	services = []

	def service_factory():
		workflow = RefundWorkflow()

		def agent_runner(customer_message, order_id):
			if customer_message == "raise":
				raise RuntimeError("injected agent failure")
			state = AgentState(
				customer_message=customer_message,
				order_id=order_id,
				case_id="CASE-RUNTIME",
			)
			state.intent = CustomerIntent.GENERAL_SUPPORT
			return state

		service = CaseApplicationService(
			cases=InMemoryCaseRepository(),
			orders=None,
			refund_workflow=workflow,
			agent_runner=agent_runner,
		)
		services.append(service)
		return service

	results = run_benchmark(
		golden_set_path=golden_path,
		case_service_factory=service_factory,
	)

	assert results[0]["status"] == "EXECUTION_ERROR"
	assert "injected agent failure" in results[0]["error"]
	assert results[0]["checks"] == {}
	assert results[1]["status"] == "PASS"
	assert len(services) == 2
	assert (
		services[0].refund_workflow.settlement.ledger
		is not services[1].refund_workflow.settlement.ledger
	)


def test_existing_processing_refund_precondition_is_seeded(tmp_path):
	golden_path = tmp_path / "golden.json"
	expected = {
		"intent": "GENERAL_SUPPORT",
		"grievances": [],
		"judgements": [],
		"settlement": {
			"decision": "AUTO_APPROVED",
			"refund_amount": 100,
			"refund_status": "PROCESSING",
			"idempotency_key": "refund:ORD-123",
		},
		"escalation": False,
		"answer": "Answer is not scored.",
	}
	golden_path.write_text(json.dumps([{
		"case_id": "CASE-PRECONDITION",
		"category": "refund_lifecycle",
		"order_id": "ORD-123",
		"customer_message": "Check my refund.",
		"preconditions": {
			"existing_refund": {
				"status": "PROCESSING",
				"amount": 100,
				"idempotency_key": "refund:ORD-123",
			}
		},
		"expected": expected,
	}]), encoding="utf-8")
	services = []

	def service_factory():
		workflow = RefundWorkflow()
		service = CaseApplicationService(
			cases=InMemoryCaseRepository(),
			orders=None,
			refund_workflow=workflow,
			agent_runner=lambda customer_message, order_id: AgentState(
				customer_message=customer_message,
				order_id=order_id,
				case_id="CASE-RUNTIME",
				settlement=SettlementDecision(
					total_refund=100,
					status=SettlementStatus.AUTO_APPROVED,
					reason="Existing processing refund.",
				),
			),
		)
		services.append(service)
		return service

	results = run_benchmark(
		golden_set_path=golden_path,
		case_service_factory=service_factory,
	)

	assert results[0]["status"] == "PASS"
	assert services[0].refund_workflow.settlement.ledger.get_refund(
		"ORD-123"
	).status == "PROCESSING"
	assert services[0].refund_workflow.refund_service.provider.calls == 0


def test_agent_settlement_uses_preconditioned_workflow_ledger(monkeypatch):
	settlements = []

	def run_agent(customer_message, order_id, settlement):
		settlements.append(settlement)
		state = AgentState(
			customer_message=customer_message,
			order_id=order_id,
			case_id="CASE-PRECONDITION",
		)
		state.settlement = settlement.settle(
			decisions=[],
			case_id=state.case_id,
			order_id=order_id,
		)
		return state

	monkeypatch.setattr(bootstrap, "run_agent", run_agent)
	container = bootstrap.build_container()
	ledger = container.refund_workflow.settlement.ledger
	refund = ledger.create_refund(
		case_id="CASE-ORIGINAL",
		order_id="ORD-123",
		amount=100,
	)
	refund.status = "PROCESSING"
	refund.idempotency_key = "refund:ORD-123"

	state = container.case_service.process_case(
		order_id="ORD-123",
		customer_message="Check my existing refund.",
	)

	assert settlements == [container.refund_workflow.settlement]
	assert state.settlement.status == SettlementStatus.PROCESSING
	assert state.settlement.total_refund == 100
	assert state.settlement.idempotency_key == "refund:ORD-123"
