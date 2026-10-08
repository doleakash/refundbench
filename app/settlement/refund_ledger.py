from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RefundRecord:
    case_id: str
    order_id: str
    amount: float
    status: str
    idempotency_key: str


class RefundLedger:
    def __init__(self, data_dir: Path | None = None):
        self._data_file = (
            data_dir / "refunds.json" if data_dir is not None else None
        )
        self._records_by_order: dict[str, RefundRecord] = {}
        if self._data_file is not None and self._data_file.exists():
            with self._data_file.open(encoding="utf-8") as file:
                self._records_by_order = {
                    order_id: RefundRecord(**record)
                    for order_id, record in json.load(file).items()
                }

    def _persist(self) -> None:
        if self._data_file is None:
            return
        self._data_file.parent.mkdir(parents=True, exist_ok=True)
        with self._data_file.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    order_id: asdict(record)
                    for order_id, record in self._records_by_order.items()
                },
                file,
                indent=2,
            )

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
        self._persist()

        return record

    def accept_refund(
        self,
        order_id: str,
    ) -> RefundRecord:

        record = self._records_by_order[order_id]

        if record.status == "PROPOSED":
            record.status = "PROCESSING"
            self._persist()

        return record

    def mark_refunded(
        self,
        order_id: str,
    ) -> RefundRecord:

        record = self._records_by_order[order_id]
        if record.status != "REFUNDED":
            record.status = "REFUNDED"
            self._persist()

        return record

    def get_refund(
        self,
        order_id: str,
    ) -> RefundRecord | None:

        return self._records_by_order.get(order_id)