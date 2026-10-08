from __future__ import annotations

from enum import Enum

from app.policy.engine import PolicyAction, PolicyDecision
from app.settlement.refund_ledger import RefundLedger


class SettlementStatus(str, Enum):
    AUTO_APPROVED = "AUTO_APPROVED"
    ESCALATE = "ESCALATE"
    PROPOSED = "PROPOSED"
    PROCESSING = "PROCESSING"
    REFUNDED = "REFUNDED"


class SettlementDecision:
    def __init__(
        self,
        total_refund: float,
        status: SettlementStatus,
        reason: str,
        idempotency_key: str | None = None,
    ):
        self.total_refund = total_refund
        self.status = status
        self.reason = reason
        self.idempotency_key = idempotency_key

    def __repr__(self):
        return (
            f"SettlementDecision("
            f"total_refund={self.total_refund}, "
            f"status={self.status}, "
            f"reason={self.reason})"
        )


class Settlement:

    REFUND_CAP = 2000.0

    def __init__(
        self,
        ledger: RefundLedger | None = None,
    ):
        self.ledger = ledger or RefundLedger()

    def settle(
        self,
        decisions: list[PolicyDecision],
        case_id: str,
        order_id: str,
    ) -> SettlementDecision:

        existing_refund = self.ledger.get_refund(order_id)
        if existing_refund is not None:
            return SettlementDecision(
                total_refund=existing_refund.amount,
                status=SettlementStatus(existing_refund.status),
                reason=(
                    f"Refund already exists for order {order_id}. "
                    f"Preserving its {existing_refund.status} status."
                ),
                idempotency_key=existing_refund.idempotency_key,
            )

        total_refund = 0.0

        for decision in decisions:
            if decision.action == PolicyAction.REFUND:
                total_refund += decision.refund_amount

        if total_refund > self.REFUND_CAP:
            return SettlementDecision(
                total_refund=total_refund,
                status=SettlementStatus.ESCALATE,
                reason=(
                    f"Refund amount ₹{total_refund:.2f} exceeds "
                    f"auto-approval cap of ₹{self.REFUND_CAP:.2f}."
                ),
            )

        refund = self.ledger.create_refund(
            case_id=case_id,
            order_id=order_id,
            amount=total_refund,
        )

        return SettlementDecision(
            total_refund=refund.amount,
            status=SettlementStatus.AUTO_APPROVED,
            reason=(
                f"Refund amount ₹{refund.amount:.2f} is within "
                f"the auto-approval cap and is now in "
                f"{refund.status} status."
            ),
        )