import pytest

from app.settlement.refund_ledger import RefundLedger


def test_created_refund_survives_reload_and_duplicate_create(tmp_path):
	ledger = RefundLedger(data_dir=tmp_path)
	created = ledger.create_refund(
		case_id="CASE-001",
		order_id="ORD-123",
		amount=100,
	)

	reloaded = RefundLedger(data_dir=tmp_path)
	existing = reloaded.get_refund("ORD-123")
	duplicate = reloaded.create_refund(
		case_id="CASE-002",
		order_id="ORD-123",
		amount=500,
	)

	assert existing is not None
	assert existing.case_id == created.case_id
	assert existing.amount == 100
	assert duplicate is existing
	assert duplicate.idempotency_key == "refund:ORD-123"


@pytest.mark.parametrize(
	("transition", "expected_status"),
	[
		("accept_refund", "PROCESSING"),
		("mark_refunded", "REFUNDED"),
	],
)
def test_refund_status_survives_reload(
	tmp_path,
	transition,
	expected_status,
):
	ledger = RefundLedger(data_dir=tmp_path)
	ledger.create_refund(
		case_id="CASE-001",
		order_id="ORD-123",
		amount=100,
	)
	if transition == "mark_refunded":
		ledger.accept_refund("ORD-123")
	getattr(ledger, transition)("ORD-123")

	reloaded = RefundLedger(data_dir=tmp_path)

	assert reloaded.get_refund("ORD-123").status == expected_status
