import json
from threading import Barrier
from pathlib import Path

from app.agent.orchestrator import run_agent
from app.agent.state import Action, AgentDecision, AgentModel
from app.domain.models import CustomerIntent, Grievance, GrievanceType
from app.evaluation.extractor import RawClaim
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


def test_agent_cases(monkeypatch):
	test_cases = load_test_cases()

	class FakeExtractor:
		model = "test-model"
		intent = CustomerIntent.REFUND_REQUEST
		raw_claims = []

		def extract(self, customer_message, case_id=None, order_id=None):
			if "delivered late" not in customer_message.lower():
				return []
			self.raw_claims = [
				RawClaim(
					raw_claim="My order was delivered late",
					type=GrievanceType.LATE_DELIVERY,
				)
			]
			return [
				Grievance(
					grievance_id="G1",
					type=GrievanceType.LATE_DELIVERY,
					claim="My order was delivered late",
					raw_claims=[
						"My order was delivered late",
					],
				)
			]

	class FakeJudge:
		model = "test-model"

		def judge(self, grievance, evidence, case_id=None, order_id=None):
			return Judgment(
				grievance_id=grievance.grievance_id,
				verdict="UPHELD",
				reason="Test evidence supports the grievance.",
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

	class FakeExtractor:
		model = "test-model"
		intent = CustomerIntent.ORDER_SUPPORT

		def extract(self, customer_message, case_id=None, order_id=None):
			return [
				Grievance(
					grievance_id="G1",
					type=GrievanceType.LATE_DELIVERY,
					claim="The order was late.",
				)
			]

	class FakeJudge:
		model = "test-model"

		def judge(self, grievance, evidence, case_id=None, order_id=None):
			return Judgment(
				grievance_id=grievance.grievance_id,
				verdict="UPHELD",
				reason="Evidence supports the grievance.",
				confidence=0.9,
			)

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
		"app.agent.action_handlers.GrievanceExtractor",
		FakeExtractor,
	)
	monkeypatch.setattr(
		"app.agent.action_handlers.Judge",
		FakeJudge,
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


def test_agent_uses_deterministic_happy_path(monkeypatch, caplog):
	caplog.set_level("INFO")

	class NoDecisionModel(AgentModel):
		def decide(self, state):
			raise AssertionError("Agent decision model must not be called")

	class FakeExtractor:
		model = "test-model"
		intent = CustomerIntent.REFUND_REQUEST
		raw_claims = [
			RawClaim(
				raw_claim="Late delivery",
				type=GrievanceType.LATE_DELIVERY,
			),
			RawClaim(
				raw_claim="Missing items",
				type=GrievanceType.MISSING_ITEMS,
			),
			RawClaim(
				raw_claim="Leaked container",
				type=GrievanceType.LEAKED_ITEM,
			),
		]

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
	assert state.intent is CustomerIntent.REFUND_REQUEST
	intent_log = next(
		record.getMessage()
		for record in caplog.records
		if record.getMessage().startswith("intent_classification ")
	)
	intent_result = json.loads(intent_log.removeprefix(
		"intent_classification "
	))
	assert intent_result["case_id"] == state.case_id
	assert intent_result["order_id"] == state.order_id
	assert intent_result["llm_extraction"]["intent"] == "REFUND_REQUEST"
	assert intent_result["llm_extraction"]["raw_claim_count"] == 3
	assert intent_result["normalization"]["normalized_grievance_count"] == 3
	assert intent_result["normalization"]["limit_exceeded"] is False
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


def test_missing_delivery_extracts_request_then_escalates_without_judges(
	monkeypatch,
):
	extractor_calls = []

	class FakeExtractor:
		model = "test-model"
		intent = CustomerIntent.REFUND_REQUEST
		raw_claims = []

		def extract(self, customer_message, case_id=None, order_id=None):
			extractor_calls.append(customer_message)
			return [
				Grievance(
					grievance_id="G1",
					type=GrievanceType.LATE_DELIVERY,
					claim="The order was late",
				)
			]

	class NoJudge:
		def __init__(self):
			raise AssertionError("Judge must not run without delivery data")

	monkeypatch.setattr(
		"app.agent.action_handlers.GrievanceExtractor",
		FakeExtractor,
	)
	monkeypatch.setattr("app.agent.action_handlers.Judge", NoJudge)

	class NoDecisionModel(AgentModel):
		def decide(self, state):
			raise AssertionError("Agent decision model must not be called")

	state = run_agent(
		customer_message="My order was late. Please check the delivery and refund me.",
		order_id="ORD-126",
		case_id="CASE-MISSING-DELIVERY",
		model=NoDecisionModel(),
	)

	assert state.actions == [
		"GET_ORDER",
		"GET_DELIVERY",
		"EXTRACT_GRIEVANCES",
		"ESCALATE",
	]
	assert extractor_calls == [
		"My order was late. Please check the delivery and refund me."
	]
	assert state.escalation_reason == (
		"Delivery record could not be found."
	)
	assert state.intent is CustomerIntent.REFUND_REQUEST
	assert [grievance.type for grievance in state.grievances] == [
		GrievanceType.LATE_DELIVERY,
	]
	assert not state.evidence
	assert not state.judgments
	assert not state.policy_decisions
	assert state.settlement.status.value == "ESCALATE"
	assert state.settlement.total_refund == 0
	assert "delivery record and required evidence are unavailable" in (
		state.response
	)
	assert not any(
		event.stage.startswith("LLM_JUDGE:")
		for event in state.timing_events
	)


def test_grievance_limit_escalates_before_evidence_and_judging(
	monkeypatch,
	caplog,
):
	caplog.set_level("INFO")
	downstream_calls = {"evidence": 0, "judge": 0}

	class FakeExtractor:
		model = "test-model"
		intent = CustomerIntent.REFUND_REQUEST
		raw_claims = [
			RawClaim(
				raw_claim="The food was spicy",
				type=GrievanceType.FOOD_QUALITY,
			),
			RawClaim(
				raw_claim="The food was burned",
				type=GrievanceType.FOOD_QUALITY,
			),
			RawClaim(
				raw_claim="The container leaked",
				type=GrievanceType.LEAKED_ITEM,
			),
			RawClaim(
				raw_claim="The food seemed strange",
				type=GrievanceType.FOOD_QUALITY,
				ambiguous=True,
			),
		]

		def extract(self, customer_message, case_id=None, order_id=None):
			return [
				Grievance(
					grievance_id="G1",
					type=GrievanceType.FOOD_QUALITY,
					claim="The food was spicy; The food was burned",
					raw_claims=[
						"The food was spicy",
						"The food was burned",
					],
				),
				Grievance(
					grievance_id="G2",
					type=GrievanceType.LEAKED_ITEM,
					claim="The container leaked",
					raw_claims=["The container leaked"],
				),
				Grievance(
					grievance_id="G3",
					type=GrievanceType.FOOD_QUALITY,
					claim="The food seemed strange",
					raw_claims=["The food seemed strange"],
				),
			]

	class FakeJudge:
		def __init__(self):
			downstream_calls["judge"] += 1

	def fake_get_evidence(*args, **kwargs):
		downstream_calls["evidence"] += 1
		raise AssertionError("Evidence must not run after claim-limit escalation")

	monkeypatch.setattr(
		"app.agent.action_handlers.GrievanceExtractor",
		FakeExtractor,
	)
	monkeypatch.setattr("app.agent.action_handlers.MAX_GRIEVANCES", 1)
	monkeypatch.setattr(
		"app.agent.action_handlers.get_evidence",
		fake_get_evidence,
	)
	monkeypatch.setattr("app.agent.action_handlers.Judge", FakeJudge)

	state = run_agent(
		customer_message="My order was late and items were missing.",
		order_id="ORD-123",
		case_id="CASE-CLAIM-LIMIT",
	)

	assert state.actions == [
		Action.GET_ORDER.value,
		Action.GET_DELIVERY.value,
		Action.EXTRACT_GRIEVANCES.value,
		Action.ESCALATE.value,
	]
	assert state.escalation_reason is not None
	assert "exceeding the limit of 1" in state.escalation_reason
	assert state.intent is CustomerIntent.REFUND_REQUEST
	assert len(state.grievances) == 3
	assert not state.evidence
	assert not state.judgments
	assert not state.policy_decisions
	assert state.settlement is None
	assert downstream_calls == {"evidence": 0, "judge": 0}

	event = next(
		record.getMessage()
		for record in caplog.records
		if record.getMessage().startswith("intent_classification ")
	)
	details = json.loads(event.removeprefix("intent_classification "))
	assert details["llm_extraction"]["intent"] == "REFUND_REQUEST"
	assert details["llm_extraction"]["raw_claim_count"] == 4
	assert details["llm_extraction"]["raw_claims"][-1]["ambiguous"]
	assert details["normalization"]["normalized_grievance_count"] == 3
	assert details["normalization"]["merged_claims"] == [{
		"grievance_id": "G1",
		"raw_claims": ["The food was spicy", "The food was burned"],
	}]
	assert details["normalization"]["grievance_limit"] == 1
	assert details["normalization"]["limit_exceeded"] is True
	assert details["normalization"]["escalation_reason"] == (
		state.escalation_reason
	)
	assert any(
		event.stage == "LLM_EXTRACT_GRIEVANCES" and event.success
		for event in state.timing_events
	)
