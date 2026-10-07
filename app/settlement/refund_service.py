from app.settlement.refund_ledger import RefundLedger
from app.settlement.refund_provider import RefundProvider


class RefundService:
    def __init__(
        self,
        ledger: RefundLedger,
        provider: RefundProvider,
    ):
        self.ledger = ledger
        self.provider = provider

    def process_refund(self, order_id: str):
        refund = self.ledger.get_refund(order_id)

        if refund is None:
            raise ValueError(
                f"No refund exists for order {order_id}"
            )

        # Refund has already been successfully processed.
        # Do not call the provider again.
        if refund.status == "REFUNDED":
            return refund

        # Customer acceptance moves the refund into PROCESSING.
        if refund.status == "PROPOSED":
            refund = self.ledger.accept_refund(order_id)

        try:
            self.provider.refund(refund)
        except Exception:
            # Keep the refund in PROCESSING so it can be
            # safely retried using the same idempotency key.
            raise

        return self.ledger.mark_refunded(order_id)