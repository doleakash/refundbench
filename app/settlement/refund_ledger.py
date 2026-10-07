from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RefundRecord:
    case_id: str
    order_id: str
    amount: float
    status: str
    idempotency_key: str


class RefundLedger:
    def __init__(self):
        self._records_by_order: dict[str, RefundRecord] = {}

    def create_refund(
        self,
        case_id: str,
        order_id: str,
        amount: float,
    ) -> RefundRecord:

        existing = self._records_by_order.get(order_id)

        if existing:
            return existing

        record = RefundRecord(
            case_id=case_id,
            order_id=order_id,
            amount=amount,
            status="PROPOSED",
            idempotency_key=f"refund:{order_id}",
        )

        self._records_by_order[order_id] = record

        return record

    def accept_refund(
        self,
        order_id: str,
    ) -> RefundRecord:

        record = self._records_by_order[order_id]

        if record.status == "PROPOSED":
            record.status = "PROCESSING"

        return record

    def mark_refunded(
        self,
        order_id: str,
    ) -> RefundRecord:

        record = self._records_by_order[order_id]
        record.status = "REFUNDED"

        return record

    def get_refund(
        self,
        order_id: str,
    ) -> RefundRecord | None:

        return self._records_by_order.get(order_id)