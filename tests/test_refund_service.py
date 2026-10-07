from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from app.settlement.refund_ledger import RefundLedger
from app.settlement.refund_provider import (
	MockRefundProvider,
	RefundProvider,
)
from app.settlement.refund_service import RefundService
from app.policy.engine import PolicyAction, PolicyDecision
from app.settlement.settlement import Settlement


def create_proposed_refund(ledger: RefundLedger):
	return ledger.create_refund(
		case_id="CASE-001",
		order_id="ORD-123",
		amount=300.0,
	)


def test_refund_lifecycle():
	ledger = RefundLedger()
	provider = MockRefundProvider()
	service = RefundService(ledger, provider)

	create_proposed_refund(ledger)

	refund = service.process_refund("ORD-123")

	assert refund.status == "REFUNDED"
	assert refund.amount == 300.0
	assert refund.idempotency_key == "refund:ORD-123"


def test_duplicate_refund_acceptance_executes_provider_once():
	settlement = Settlement()
	settlement.settle(
		decisions=[
			PolicyDecision(
				grievance_id="G1",
				action=PolicyAction.REFUND,
				refund_amount=300.0,
				reason="Refund approved.",
			)
		],
		case_id="CASE-001",
		order_id="ORD-123",
	)
	provider = MockRefundProvider()
	service = RefundService(settlement.ledger, provider)

	first = service.process_refund("ORD-123")
	second = service.process_refund("ORD-123")

	assert provider.calls == 1
	assert first is second
	assert first.idempotency_key == second.idempotency_key == "refund:ORD-123"
	assert first.status == second.status == "REFUNDED"


def test_concurrent_duplicate_acceptance_executes_provider_once():
	ledger = RefundLedger()
	create_proposed_refund(ledger)
	provider = MockRefundProvider()
	service = RefundService(ledger, provider)
	provider_boundary = Barrier(2, timeout=5)
	mock_refund = provider.refund

	def synchronized_refund(refund):
		provider_boundary.wait()
		return mock_refund(refund)

	provider.refund = synchronized_refund

	with ThreadPoolExecutor(max_workers=2) as executor:
		futures = [
			executor.submit(service.process_refund, "ORD-123")
			for _ in range(2)
		]
		first, second = [future.result(timeout=10) for future in futures]

	assert first is second
	assert first.idempotency_key == second.idempotency_key == "refund:ORD-123"
	assert first.status == second.status == "REFUNDED"
	assert ledger.get_refund("ORD-123") is first
	assert provider.calls == 1


def test_ambiguous_provider_timeout_reuses_recorded_operation():
	ledger = RefundLedger()
	provider = MockRefundProvider()
	requests = 0
	mock_refund = provider.refund

	def timeout_after_provider_success(refund):
		nonlocal requests
		requests += 1
		mock_refund(refund)
		if requests == 1:
			raise TimeoutError("Provider response timed out")

	provider.refund = timeout_after_provider_success
	service = RefundService(ledger, provider)
	create_proposed_refund(ledger)

	with pytest.raises(TimeoutError, match="Provider response timed out"):
		service.process_refund("ORD-123")

	pending_refund = ledger.get_refund("ORD-123")
	assert pending_refund is not None
	assert pending_refund.status == "PROCESSING"

	refund = service.process_refund("ORD-123")

	assert requests == 2
	assert provider.calls == 1
	assert refund.status == "REFUNDED"
	assert refund.idempotency_key == "refund:ORD-123"


class FailingRefundProvider(RefundProvider):
	def refund(self, refund):
		raise RuntimeError("Payment provider timeout")


def test_provider_failure_keeps_refund_processing():
	ledger = RefundLedger()
	provider = FailingRefundProvider()
	service = RefundService(ledger, provider)

	create_proposed_refund(ledger)

	with pytest.raises(RuntimeError, match="Payment provider timeout"):
		service.process_refund("ORD-123")

	refund = ledger.get_refund("ORD-123")

	assert refund is not None
	assert refund.status == "PROCESSING"
	assert refund.idempotency_key == "refund:ORD-123"


def test_retry_after_provider_failure_refunds_successfully():
	ledger = RefundLedger()

	class FlakyRefundProvider(RefundProvider):
		def __init__(self):
			self.calls = 0
			self.idempotency_keys = []

		def refund(self, refund):
			self.calls += 1
			self.idempotency_keys.append(refund.idempotency_key)

			if self.calls == 1:
				raise RuntimeError("Payment provider timeout")

	provider = FlakyRefundProvider()
	service = RefundService(ledger, provider)

	create_proposed_refund(ledger)

	with pytest.raises(RuntimeError):
		service.process_refund("ORD-123")

	# Second attempt succeeds.
	def successful_refund(refund):
		provider.calls += 1
		provider.idempotency_keys.append(refund.idempotency_key)

	provider.refund = successful_refund

	refund = service.process_refund("ORD-123")

	assert refund.status == "REFUNDED"
	assert provider.calls == 2
	assert provider.idempotency_keys == [
		"refund:ORD-123",
		"refund:ORD-123",
	]


def test_same_order_does_not_create_duplicate_refund():
	ledger = RefundLedger()

	first = ledger.create_refund(
		case_id="CASE-001",
		order_id="ORD-123",
		amount=300.0,
	)

	second = ledger.create_refund(
		case_id="CASE-002",
		order_id="ORD-123",
		amount=500.0,
	)

	assert first is second
	assert second.amount == 300.0
	assert second.idempotency_key == "refund:ORD-123"
