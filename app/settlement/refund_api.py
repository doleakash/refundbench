from app.settlement.refund_workflow import RefundWorkflow


class RefundAPI:
    def __init__(self, workflow: RefundWorkflow):
        self.workflow = workflow

    def accept_refund(self, order_id: str):
        refund = self.workflow.accept_refund(
            order_id=order_id
        )

        return {
            "order_id": refund.order_id,
            "status": refund.status,
            "amount": refund.amount,
            "idempotency_key": refund.idempotency_key,
        }