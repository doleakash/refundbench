from __future__ import annotations

from app.settlement.refund_provider import MockRefundProvider
from app.settlement.refund_ledger import RefundRecord
from app.settlement.refund_service import RefundService
from app.settlement.settlement import Settlement


class RefundWorkflow:
    def __init__(
        self,
        settlement: Settlement | None = None,
        refund_service: RefundService | None = None,
    ):
        self.settlement = settlement or Settlement()
        self.refund_service = refund_service or RefundService(
            ledger=self.settlement.ledger,
            provider=MockRefundProvider(),
        )

    def accept_refund(self, order_id: str) -> RefundRecord:
        refund = self.refund_service.process_refund(
            order_id=order_id,
        )

        return refund

    def create_refund_proposal(
        self,
        case_id: str,
        order_id: str,
        amount: float,
    ) -> RefundRecord:
        return self.settlement.ledger.create_refund(
            case_id=case_id,
            order_id=order_id,
            amount=amount,
        )