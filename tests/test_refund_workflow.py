from app.policy.engine import PolicyAction, PolicyDecision
from app.settlement.refund_workflow import RefundWorkflow
from app.settlement.refund_api import RefundAPI


def test_refund_workflow():
    workflow = RefundWorkflow()

    decisions = [
        PolicyDecision(
            grievance_id="G1",
            action=PolicyAction.REFUND,
            refund_amount=300.0,
            reason="Late delivery grievance upheld.",
        )
    ]

    settlement = workflow.settlement.settle(
        decisions=decisions,
        case_id="CASE-001",
        order_id="ORD-123",
    )

    assert settlement.total_refund == 300.0

    refund = workflow.settlement.ledger.get_refund(
        "ORD-123"
    )

    assert refund is not None
    assert refund.status == "PROPOSED"

    refunded = workflow.accept_refund("ORD-123")

    assert refunded.status == "REFUNDED"
    assert refunded.amount == 300.0
    assert refunded.idempotency_key == "refund:ORD-123"


def test_settlement_preserves_existing_processing_refund():
    workflow = RefundWorkflow()
    existing = workflow.settlement.ledger.create_refund(
        case_id="CASE-ORIGINAL",
        order_id="ORD-123",
        amount=100.0,
    )
    workflow.settlement.ledger.accept_refund("ORD-123")

    result = workflow.settlement.settle(
        decisions=[
            PolicyDecision(
                grievance_id="G1",
                action=PolicyAction.REFUND,
                refund_amount=500.0,
                reason="Duplicate request.",
            )
        ],
        case_id="CASE-DUPLICATE",
        order_id="ORD-123",
    )

    assert result.status.value == "PROCESSING"
    assert result.total_refund == 100.0
    assert result.idempotency_key == "refund:ORD-123"
    assert workflow.settlement.ledger.get_refund("ORD-123") is existing
    assert existing.status == "PROCESSING"
    assert workflow.refund_service.provider.calls == 0



def test_accept_refund_api():
    workflow = RefundWorkflow()

    workflow.settlement.settle(
        decisions=[
            PolicyDecision(
                grievance_id="G1",
                action=PolicyAction.REFUND,
                refund_amount=300.0,
                reason="Late delivery grievance upheld.",
            )
        ],
        case_id="CASE-001",
        order_id="ORD-123",
    )

    api = RefundAPI(workflow)

    result = api.accept_refund("ORD-123")

    assert result["order_id"] == "ORD-123"
    assert result["status"] == "REFUNDED"
    assert result["amount"] == 300.0
    assert result["idempotency_key"] == "refund:ORD-123"