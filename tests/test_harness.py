import pytest

from app.agent.harness import AgentHarness
from app.agent.state import Action, AgentDecision, AgentState


def test_cannot_get_delivery_before_order():
	state = AgentState(
		customer_message="Where is my order?",
		order_id="ORD-123",
		case_id="CASE-001",
	)

	harness = AgentHarness()

	with pytest.raises(ValueError):
		harness.validate(
			state,
			AgentDecision(action=Action.GET_DELIVERY),
		)


def test_cannot_stop_without_resolution():
	state = AgentState(
		customer_message="Where is my order?",
		order_id="ORD-123",
		case_id="CASE-001",
	)

	harness = AgentHarness()

	with pytest.raises(ValueError):
		harness.validate(
			state,
			AgentDecision(action=Action.STOP),
		)


def test_harness_blocks_invalid_agent_decision():
	state = AgentState(
		customer_message="Where is my order?",
		order_id="ORD-123",
		case_id="CASE-001",
	)

	harness = AgentHarness()

	bad_decision = AgentDecision(
		action=Action.GET_DELIVERY
	)

	with pytest.raises(
		ValueError,
		match="Cannot get delivery before retrieving the order",
	):
		harness.validate(state, bad_decision)
