import json
from threading import Barrier
from pathlib import Path

from app.agent.orchestrator import run_agent
from app.agent.state import Action, AgentDecision, AgentModel
from app.domain.models import Grievance, GrievanceType
from app.evaluation.judge import Judgment

DATA_FILE = (
	Path(__file__).resolve().parent.parent
	/ "data"
	/ "agent_test_cases.json"
)


class FullPipelineModel(AgentModel):
	def decide(self, state):
		if not state.actions:
			return AgentDecision(action=Action.GET_ORDER)

		last_action = state.actions[-1]

		if last_action == Action.GET_ORDER.value:
			if state.order is None:
				return AgentDecision(action=Action.ESCALATE)

			return AgentDecision(action=Action.GET_DELIVERY)

		if last_action == Action.GET_DELIVERY.value:
			return AgentDecision(action=Action.EXTRACT_GRIEVANCES)

		if last_action == Action.EXTRACT_GRIEVANCES.value:
			if not state.grievances:
				return AgentDecision(action=Action.SETTLE)

			return AgentDecision(action=Action.GET_EVIDENCE)

		if last_action == Action.GET_EVIDENCE.value:
			return AgentDecision(action=Action.RUN_JUDGES)

		if last_action == Action.RUN_JUDGES.value:
			return AgentDecision(action=Action.BUILD_CONSENSUS)

		if last_action == Action.BUILD_CONSENSUS.value:
			return AgentDecision(action=Action.APPLY_POLICY)

		if last_action == Action.APPLY_POLICY.value:
			return AgentDecision(action=Action.SETTLE)

		if last_action == Action.SETTLE.value:
			return AgentDecision(action=Action.STOP)

		return AgentDecision(action=Action.STOP)


class FailingOrderToolModel(AgentModel):
	def decide(self, state):
		if state.tool_errors:
			return AgentDecision(action=Action.ESCALATE)

		return AgentDecision(action=Action.GET_ORDER)


class RetryOrderToolModel(AgentModel):
	def __init__(self):
		self.calls = 0

	def decide(self, state):
		self.calls += 1

		if self.calls == 1:
			return AgentDecision(action=Action.GET_ORDER)

		if self.calls == 2:
			return AgentDecision(action=Action.GET_DELIVERY)

		if self.calls == 3:
			return AgentDecision(action=Action.EXTRACT_GRIEVANCES)

		if self.calls == 4:
			return AgentDecision(action=Action.GET_EVIDENCE)

		if self.calls == 5:
			return AgentDecision(action=Action.RUN_JUDGES)

		if self.calls == 6:
			return AgentDecision(action=Action.BUILD_CONSENSUS)

		if self.calls == 7:
			return AgentDecision(action=Action.APPLY_POLICY)

		if self.calls == 8:
			return AgentDecision(action=Action.SETTLE)

		return AgentDecision(action=Action.STOP)


def load_test_cases():
	with open(DATA_FILE) as f:
		return json.load(f)


def test_agent_cases():
	test_cases = load_test_cases()

	for case in test_cases:
		fake_model = FullPipelineModel()

		state = run_agent(
			customer_message=case["customer_message"],
			order_id=case["order_id"],
			case_id=case["case_id"],
			model=fake_model,
		)

		assert state.actions
		assert state.actions[-1] in {
			Action.STOP.value,
			Action.ESCALATE.value,
		}

		if Action.SETTLE.value in state.actions:
			assert state.settlement is not None


def test_agent_captures_tool_failure(monkeypatch):
	def failing_get_order(order_id):
		raise RuntimeError("Order service unavailable")

	monkeypatch.setattr(
		"app.agent.action_handlers.get_order",
		failing_get_order,
	)

	state = run_agent(
		customer_message="Where is my order?",
		order_id="ORD-123",
		case_id="CASE-FAILURE",
		model=FailingOrderToolModel(),
		max_iterations=2,
	)

	assert len(state.tool_errors) == 2
	assert "GET_ORDER" in state.tool_errors[0]
	assert "Order service unavailable" in state.tool_errors[0]
	assert state.tool_retry_count["GET_ORDER"] == 1


def test_agent_retries_transient_tool_failure(monkeypatch):
	calls = 0

	original_get_order = __import__(
		"app.agent.action_handlers",
		fromlist=["get_order"],
	).get_order

	def flaky_get_order(order_id):
		nonlocal calls
		calls += 1

		if calls == 1:
			raise RuntimeError("Order service timeout")

		return original_get_order(order_id)

	monkeypatch.setattr(
		"app.agent.action_handlers.get_order",
		flaky_get_order,
	)

	state = run_agent(
		customer_message="Where is my order?",
		order_id="ORD-123",
		case_id="CASE-RETRY",
		model=RetryOrderToolModel(),
	)

	assert calls == 2
	assert state.tool_retry_count["GET_ORDER"] == 1
	assert state.order is not None
	assert state.settlement is not None


def test_agent_rejects_repeated_successful_action_but_allows_retry(monkeypatch):
	calls = 0
	original_get_order = __import__(
		"app.agent.action_handlers",
		fromlist=["get_order"],
	).get_order

	def flaky_get_order(order_id):
		nonlocal calls
		calls += 1

		if calls == 1:
			raise RuntimeError("Order service timeout")

		return original_get_order(order_id)

	monkeypatch.setattr(
		"app.agent.action_handlers.get_order",
		flaky_get_order,
	)
	monkeypatch.setattr(
		"app.agent.orchestrator._get_deterministic_next_action",
		lambda state, completed_action: Action.GET_ORDER,
	)

	state = run_agent(
		customer_message="Where is my order?",
		order_id="ORD-123",
		case_id="CASE-REPEATED-ACTION",
		model=RetryOrderToolModel(),
	)

	assert calls == 2
	assert state.tool_retry_count["GET_ORDER"] == 1
	assert state.actions == [
		Action.GET_ORDER.value,
		Action.GET_ORDER.value,
		Action.ESCALATE.value,
	]
	assert any(
		"Repeated action GET_ORDER was rejected" in observation
		for observation in state.observations
	)
	assert state.escalation_reason == (
		"Repeated action GET_ORDER cannot be safely executed."
	)


def test_agent_uses_deterministic_happy_path(monkeypatch):
	class NoDecisionModel(AgentModel):
		def decide(self, state):
			raise AssertionError("Agent decision model must not be called")

	class FakeExtractor:
		model = "test-model"

		def extract(self, customer_message, case_id=None, order_id=None):
			return [
				Grievance(
					grievance_id=f"G{index}",
					type=grievance_type,
					claim=customer_message,
				)
				for index, grievance_type in enumerate(
					[
						GrievanceType.LATE_DELIVERY,
						GrievanceType.MISSING_ITEMS,
						GrievanceType.LEAKED_ITEM,
					],
					start=1,
				)
			]

	class FakeJudge:
		model = "test-model"
		all_judges_started = Barrier(9, timeout=5)

		def judge(self, grievance, evidence, case_id=None, order_id=None):
			self.all_judges_started.wait()
			return Judgment(
				grievance_id=grievance.grievance_id,
				verdict="UPHELD",
				reason="Evidence supports the grievance.",
				confidence=0.9,
			)

	monkeypatch.setattr(
		"app.agent.action_handlers.GrievanceExtractor",
		FakeExtractor,
	)
	monkeypatch.setattr(
		"app.agent.action_handlers.Judge",
		FakeJudge,
	)

	state = run_agent(
		customer_message="Late delivery, missing items, leaked container.",
		order_id="ORD-123",
		case_id="CASE-DETERMINISTIC",
		model=NoDecisionModel(),
	)

	assert state.actions == [
		"GET_ORDER",
		"GET_DELIVERY",
		"EXTRACT_GRIEVANCES",
		"GET_EVIDENCE",
		"RUN_JUDGES",
		"BUILD_CONSENSUS",
		"APPLY_POLICY",
		"SETTLE",
		"STOP",
	]
	assert state.settlement.total_refund == 740.0
	assert sum(
		event.stage == "LLM_EXTRACT_GRIEVANCES"
		for event in state.timing_events
	) == 1
	assert sum(
		event.stage.startswith("LLM_JUDGE:")
		for event in state.timing_events
	) == 9
	assert not any(
		event.stage == "LLM_AGENT_DECISION"
		for event in state.timing_events
	)
	assert all(
		event.case_id == state.case_id
		and event.order_id == state.order_id
		for event in state.timing_events
	)

	from app.application.cases.repository import InMemoryCaseRepository
	from app.application.cases.service import CaseApplicationService
	from app.infrastructure.order_repository import JSONOrderRepository
	from app.settlement.refund_workflow import RefundWorkflow

	workflow = RefundWorkflow()
	case_repository = InMemoryCaseRepository()
	case_repository.save(state)
	service = CaseApplicationService(
		cases=case_repository,
		orders=JSONOrderRepository(
			Path(__file__).resolve().parent.parent / "data"
		),
		refund_workflow=workflow,
		agent_runner=lambda **kwargs: state,
	)
	workflow.create_refund_proposal(
		case_id=state.case_id,
		order_id=state.order_id,
		amount=state.settlement.total_refund,
	)

	refund_result = service.accept_refund(state.case_id)

	assert refund_result.status == "REFUNDED"
	refund_event = state.timing_events[-1]
	assert refund_event.stage == "REFUND_ACCEPTANCE:REFUNDED"
	assert refund_event.success
	assert all(
		event.case_id == state.case_id
		and event.order_id == state.order_id
		for event in state.timing_events
	)


def test_missing_delivery_escalates_without_llm_calls():
	class NoDecisionModel(AgentModel):
		def decide(self, state):
			raise AssertionError("Agent decision model must not be called")

	state = run_agent(
		customer_message="My order was late.",
		order_id="ORD-126",
		case_id="CASE-MISSING-DELIVERY",
		model=NoDecisionModel(),
	)

	assert state.actions == [
		"GET_ORDER",
		"GET_DELIVERY",
		"ESCALATE",
	]
	assert state.escalation_reason == (
		"Delivery record could not be found."
	)
	assert not state.grievances
	assert not any(event.model is not None for event in state.timing_events)
