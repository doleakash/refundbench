from app.policy.engine import PolicyAction, PolicyDecision
from app.settlement.settlement import SettlementDecision, SettlementStatus


def generate_grievance_response(
    decisions: list[PolicyDecision],
    settlement: SettlementDecision,
) -> str:

    if settlement.status == SettlementStatus.ESCALATE:
        return (
            "Your refund request requires additional review. "
            f"The calculated refund is ₹{settlement.total_refund:.0f}, "
            "but it has not been approved or issued."
        )

    messages = []

    if settlement.total_refund > 0:
        messages.append(
            f"A refund of ₹{settlement.total_refund:.0f} "
            "has been approved."
        )

    for decision in decisions:
        if decision.action == PolicyAction.REFUND:
            messages.append(
                f"Grievance {decision.grievance_id}: "
                f"₹{decision.refund_amount:.0f} approved. "
                f"{decision.reason}"
            )

        elif decision.action == PolicyAction.REQUEST_EVIDENCE:
            messages.append(
                f"Grievance {decision.grievance_id}: "
                "We need additional evidence to review this claim."
            )

        elif decision.action == PolicyAction.ESCALATE:
            messages.append(
                f"Grievance {decision.grievance_id}: "
                "This claim requires human review."
            )

        elif decision.action == PolicyAction.NO_REFUND:
            messages.append(
                f"Grievance {decision.grievance_id}: "
                "No refund was approved for this claim."
            )

    if not messages:
        return "Your request has been reviewed. No refund was approved."

    return " ".join(messages)