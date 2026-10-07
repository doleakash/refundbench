from app.settlement.refund_api import RefundAPI
from app.settlement.refund_workflow import RefundWorkflow


def test_accept_refund_api():
    workflow = RefundWorkflow()

    workflow.settlement.settle(
        decisions=[
            # Use the same PolicyDecision setup
            # from test_refund_workflow.py
        ],
        case_id="CASE-001",
        order_id="ORD-123",
    )

    api = RefundAPI(workflow)

    result = api.accept_refund("ORD-123")

    assert result["order_id"] == "ORD-123"
    assert result["status"] == "REFUNDED"