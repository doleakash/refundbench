from app.response.generator import generate_grievance_response
from app.settlement.settlement import SettlementDecision, SettlementStatus


def test_processing_refund_response_preserves_existing_refund_details():
    settlement = SettlementDecision(
        total_refund=100,
        status=SettlementStatus.PROCESSING,
        reason="An existing refund is processing.",
        idempotency_key="refund:ORD-123",
    )

    response = generate_grievance_response([], settlement)

    assert "existing refund of ₹100 remains in progress" in response
    assert "No duplicate refund was created" in response
    assert "refund:ORD-123" in response
    assert "has been approved" not in response


def test_newly_approved_refund_response_is_unchanged():
    settlement = SettlementDecision(
        total_refund=100,
        status=SettlementStatus.AUTO_APPROVED,
        reason="Approved.",
    )

    assert generate_grievance_response([], settlement) == (
        "A refund of ₹100 has been approved."
    )
