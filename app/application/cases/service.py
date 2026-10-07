import logging
from time import perf_counter
from typing import Callable

from app.agent.state import AgentState
from app.application.cases.errors import (
	CaseNotFound,
	InvalidCaseRequest,
	RefundNotAcceptable,
)
from app.application.cases.repository import CaseRepository, OrderRepository
from app.infrastructure.observability import TimingEvent, log_timing_event
from app.settlement.refund_ledger import RefundRecord
from app.settlement.refund_workflow import RefundWorkflow
from app.settlement.settlement import SettlementStatus

logger = logging.getLogger(__name__)
AgentRunner = Callable[..., AgentState]


class CaseApplicationService:
	def __init__(
		self,
		cases: CaseRepository,
		orders: OrderRepository,
		refund_workflow: RefundWorkflow,
		agent_runner: AgentRunner,
	):
		self.cases = cases
		self.orders = orders
		self.refund_workflow = refund_workflow
		self.agent_runner = agent_runner

	def list_order_ids(self) -> list[str]:
		return self.orders.list_order_ids()

	def process_case(
		self,
		order_id: str,
		customer_message: str,
	) -> AgentState:
		order_id = order_id.strip()
		customer_message = customer_message.strip()
		if not order_id or not customer_message:
			raise InvalidCaseRequest(
				"Order ID and customer complaint are required."
			)

		state = self.agent_runner(
			customer_message=customer_message,
			order_id=order_id,
		)
		self.cases.save(state)

		if (
			state.settlement is not None
			and state.settlement.status == SettlementStatus.AUTO_APPROVED
			and state.settlement.total_refund > 0
		):
			self.refund_workflow.create_refund_proposal(
				case_id=state.case_id,
				order_id=state.order_id,
				amount=state.settlement.total_refund,
			)

		return state

	def get_case(self, case_id: str) -> AgentState:
		state = self.cases.get(case_id)
		if state is None:
			raise CaseNotFound(f"Case {case_id} was not found.")
		return state

	def accept_refund(self, case_id: str) -> RefundRecord:
		state = self.get_case(case_id)
		settlement = state.settlement
		if (
			settlement is None
			or settlement.status != SettlementStatus.AUTO_APPROVED
			or settlement.total_refund <= 0
		):
			raise RefundNotAcceptable(
				"This case has no approved refund proposal to accept."
			)

		started_at = perf_counter()
		try:
			refund = self.refund_workflow.accept_refund(state.order_id)
		except Exception:
			event = TimingEvent(
				case_id=state.case_id,
				order_id=state.order_id,
				stage="REFUND_ACCEPTANCE",
				duration_ms=(perf_counter() - started_at) * 1000,
				success=False,
			)
			state.timing_events.append(event)
			log_timing_event(event)
			logger.exception(
				"Refund acceptance failed for case %s",
				case_id,
			)
			raise

		event = TimingEvent(
			case_id=state.case_id,
			order_id=state.order_id,
			stage=f"REFUND_ACCEPTANCE:{refund.status}",
			duration_ms=(perf_counter() - started_at) * 1000,
			success=True,
		)
		state.timing_events.append(event)
		log_timing_event(event)
		return refund
