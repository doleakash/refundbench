from app.agent.state import AgentState
from app.interfaces.api.schemas import (
	CaseResult,
	EvidenceResult,
	GrievanceResult,
	JudgmentResult,
	PolicyDecisionResult,
	SettlementResult,
)
from app.settlement.settlement import SettlementStatus


def case_to_response(state: AgentState) -> CaseResult:
	settlement = state.settlement
	if settlement is not None:
		status = (
			"ESCALATED"
			if settlement.status == SettlementStatus.ESCALATE
			else settlement.status.value
		)
		settlement_result = SettlementResult(
			status=settlement.status.value,
			amount=settlement.total_refund,
			reason=settlement.reason,
			idempotency_key=settlement.idempotency_key,
		)
	else:
		settlement_result = None
		status = (
			"ESCALATED"
			if state.escalation_reason
			or (state.actions and state.actions[-1] == "ESCALATE")
			or "Tool failure requires human escalation."
			in state.observations
			else "INCOMPLETE"
		)

	grievances = []
	for grievance in state.grievances:
		grievance_id = grievance.grievance_id
		evidence = state.evidence.get(grievance_id)
		decision = state.policy_decisions.get(grievance_id)
		grievances.append(
			GrievanceResult(
				grievance_id=grievance_id,
				type=grievance.type.value,
				claim=grievance.claim,
				raw_claims=grievance.raw_claims,
				evidence=(
					EvidenceResult(
						source=evidence.source,
						facts=evidence.facts,
						requires_customer_input=(
							evidence.requires_customer_input
						),
					)
					if evidence is not None
					else None
				),
				judgments=[
					JudgmentResult(
						verdict=judgment.verdict,
						reason=judgment.reason,
						confidence=judgment.confidence,
					)
					for judgment in state.judgments.get(grievance_id, [])
				],
				consensus=state.consensus.get(grievance_id),
				policy_decision=(
					PolicyDecisionResult(
						action=decision.action.value,
						refund_amount=decision.refund_amount,
						reason=decision.reason,
					)
					if decision is not None
					else None
				),
			)
		)

	return CaseResult(
		case_id=state.case_id,
		order_id=state.order_id,
		intent=state.intent.value,
		status=status,
		response=state.response,
		settlement=settlement_result,
		grievances=grievances,
		actions=state.actions,
		observations=state.observations,
		tool_errors=state.tool_errors,
		escalation_reason=state.escalation_reason,
		timing_events=state.timing_events,
	)
