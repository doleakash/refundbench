from abc import ABC, abstractmethod
from threading import Lock

from app.settlement.refund_ledger import RefundRecord


class RefundProvider(ABC):

	@abstractmethod
	def refund(self, refund: RefundRecord) -> None:
		raise NotImplementedError


class MockRefundProvider(RefundProvider):

	def __init__(self):
		self.calls = 0
		self._refunds_by_idempotency_key: dict[str, RefundRecord] = {}
		self._lock = Lock()

	def refund(self, refund: RefundRecord) -> None:
		with self._lock:
			if refund.idempotency_key in self._refunds_by_idempotency_key:
				return

			self._refunds_by_idempotency_key[refund.idempotency_key] = refund
			self.calls += 1
		# Simulate successful payment-provider refund.
		return None
