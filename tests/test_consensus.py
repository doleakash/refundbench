from types import SimpleNamespace

import pytest

from app.agent.orchestrator import classify_terminal_outcome
from app.agent.state import AgentState
from app.domain.models import Evidence, Grievance, GrievanceType
from app.evaluation.consensus import Consensus
from app.evaluation.judge import Judgment
from app.policy.engine import Policy, PolicyAction
from app.settlement.settlement import Settlement, SettlementStatus
from benchmark.runner import _actual_outcome


def judgment(verdict: str) -> Judgment:
	return Judgment(
		grievance_id="G1",
		verdict=verdict,
		reason="test vote",
		confidence=1.0,
	)


@pytest.mark.parametrize(
	("votes", "expected"),
	[
		(["UPHELD", "UPHELD", "REJECTED"], "UPHELD"),
		(
			["UPHELD", "REJECTED", "INSUFFICIENT_EVIDENCE"],
			"ESCALATE",
		),
	],
)
def test_consensus_requires_two_matching_judges(votes, expected):
	assert Consensus().decide([judgment(vote) for vote in votes]) == expected


def test_no_majority_escalates_without_a_refund_and_records_current_outcome():
	grievance = Grievance(
		grievance_id="G1",
		type=GrievanceType.MISSING_ITEMS,
		claim="An item may be missing.",
	)
	evidence = Evidence(
		grievance_id="G1",
		source="ORDER_DELIVERY_DATA",
		facts=["Item-level evidence is inconclusive."],
	)
	consensus = Consensus().decide([
		judgment("UPHELD"),
		judgment("REJECTED"),
		judgment("INSUFFICIENT_EVIDENCE"),
	])
	decision = Policy().decide(grievance, evidence, consensus)
	assert consensus == "ESCALATE"
	assert decision.action is PolicyAction.ESCALATE
	assert decision.refund_amount == 0

	settlement_service = Settlement()
	settlement = settlement_service.settle(
		decisions=[decision],
		case_id="CASE-CONSENSUS-ESCALATE",
		order_id="ORD-CONSENSUS-ESCALATE",
	)
	assert settlement.total_refund == 0
	assert settlement.status is SettlementStatus.AUTO_APPROVED
	zero_value_record = settlement_service.ledger.get_refund(
		"ORD-CONSENSUS-ESCALATE"
	)
	assert zero_value_record.amount == 0
	assert zero_value_record.status == "PROPOSED"

	state = AgentState(
		customer_message="An item may be missing.",
		order_id="ORD-CONSENSUS-ESCALATE",
		case_id="CASE-CONSENSUS-ESCALATE",
		actions=["STOP"],
		consensus={"G1": consensus},
		policy_decisions={"G1": decision},
		settlement=settlement,
	)
	state.outcome = classify_terminal_outcome(state)
	actual = _actual_outcome(
		state,
		SimpleNamespace(
			refund_workflow=SimpleNamespace(settlement=settlement_service)
		),
	)
	assert actual.escalation is True
	assert actual.settlement.refund_amount == 0
	assert actual.settlement.refund_status == "PROPOSED"
	# Current terminal classification keys off the STOP action and settlement
	# status, so a policy-level escalation still has a RESOLVED workflow outcome.
	assert actual.outcome == "RESOLVED"
