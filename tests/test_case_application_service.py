import json
from types import SimpleNamespace

from app.agent.state import AgentState
from app.application.cases.repository import InMemoryCaseRepository
from app.application.cases.service import CaseApplicationService
from app.infrastructure.order_repository import JSONOrderRepository
from app.interfaces.api.routes import create_api_app
from app.settlement.refund_workflow import RefundWorkflow
from app.settlement.settlement import SettlementDecision, SettlementStatus
from fastapi.testclient import TestClient


def test_case_service_stores_case_and_owns_refund_lifecycle(tmp_path):
	orders_file = tmp_path / "orders.json"
	orders_file.write_text(
		json.dumps([{"order_id": "ORD-123"}]),
		encoding="utf-8",
	)
	state = AgentState(
		customer_message="Order was late.",
		order_id="ORD-123",
		case_id="CASE-APP",
		settlement=SettlementDecision(
			total_refund=300.0,
			status=SettlementStatus.AUTO_APPROVED,
			reason="Approved.",
		),
	)
	runner_calls = []

	def run_agent(**kwargs):
		runner_calls.append(kwargs)
		return state

	cases = InMemoryCaseRepository()
	workflow = RefundWorkflow()
	service = CaseApplicationService(
		cases=cases,
		orders=JSONOrderRepository(tmp_path),
		refund_workflow=workflow,
		agent_runner=run_agent,
	)

	processed = service.process_case(
		order_id=" ORD-123 ",
		customer_message=" Order was late. ",
	)

	assert processed is state
	assert service.get_case("CASE-APP") is state
	assert runner_calls == [
		{
			"customer_message": "Order was late.",
			"order_id": "ORD-123",
		}
	]
	assert service.list_order_ids() == ["ORD-123"]
	assert workflow.settlement.ledger.get_refund("ORD-123").status == "PROPOSED"

	refund = service.accept_refund("CASE-APP")

	assert refund.status == "REFUNDED"
	assert refund.idempotency_key == "refund:ORD-123"
	assert state.timing_events[-1].stage == "REFUND_ACCEPTANCE:REFUNDED"


def test_case_api_preserves_case_and_refund_response_contract(
	tmp_path,
	monkeypatch,
):
	(tmp_path / "orders.json").write_text(
		json.dumps([{"order_id": "ORD-123"}]),
		encoding="utf-8",
	)
	state = AgentState(
		customer_message="Order was late.",
		order_id="ORD-123",
		case_id="CASE-API",
		settlement=SettlementDecision(
			total_refund=300.0,
			status=SettlementStatus.AUTO_APPROVED,
			reason="Approved.",
		),
	)
	runner_calls = []

	def run_agent(**kwargs):
		runner_calls.append(kwargs)
		return state

	service = CaseApplicationService(
		cases=InMemoryCaseRepository(),
		orders=JSONOrderRepository(tmp_path),
		refund_workflow=RefundWorkflow(),
		agent_runner=run_agent,
	)
	monkeypatch.setattr(
		"app.application.cases.service.get_settings",
		lambda: SimpleNamespace(max_customer_message_length=50),
	)
	client = TestClient(create_api_app(service))

	assert client.get("/orders").json() == ["ORD-123"]

	created = client.post(
		"/cases",
		json={
			"order_id": "ORD-123",
			"customer_message": "Order was late.",
		},
	)
	assert created.status_code == 200
	assert created.json()["case_id"] == "CASE-API"
	assert created.json()["intent"] == "GENERAL_SUPPORT"
	assert client.get("/cases/CASE-API").json() == created.json()
	assert len(runner_calls) == 1

	too_long = client.post(
		"/cases",
		json={
			"order_id": "ORD-123",
			"customer_message": "x" * 51,
		},
	)
	assert too_long.status_code == 422
	assert "maximum length of 50 characters" in too_long.json()["detail"]
	assert len(runner_calls) == 1

	accepted = client.post("/cases/CASE-API/refund/accept")
	assert accepted.status_code == 200
	assert accepted.json() == {
		"case_id": "CASE-API",
		"order_id": "ORD-123",
		"status": "REFUNDED",
		"amount": 300.0,
		"idempotency_key": "refund:ORD-123",
	}
