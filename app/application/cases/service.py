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
from app.domain.models import CustomerIntent
from app.infrastructure.observability import TimingEvent, log_timing_event
from app.settlement.refund_ledger import RefundRecord
from app.settlement.refund_workflow import RefundWorkflow
from app.settlement.settlement import SettlementDecision, SettlementStatus
from config.settings import get_settings

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
		if not order_id or not customer_message.strip():
			raise InvalidCaseRequest(
				"Order ID and customer complaint are required."
			)
		max_length = get_settings().max_customer_message_length
		if len(customer_message) > max_length:
			raise InvalidCaseRequest(
				"Customer message exceeds the maximum length "
				f"of {max_length} characters."
			)
		customer_message = customer_message.strip()

		existing_refund = self.refund_workflow.settlement.ledger.get_refund(
			order_id
		)
		state = self.agent_runner(
			customer_message=customer_message,
			order_id=order_id,
		)

		if (
			existing_refund is not None
			and state.intent == CustomerIntent.REFUND_REQUEST
			and state.settlement is not None
		):
			state.settlement = SettlementDecision(
				total_refund=existing_refund.amount,
				status=SettlementStatus(existing_refund.status),
				reason=(
					f"An existing refund is "
					f"{existing_refund.status.lower()} for this order. "
					"No duplicate refund was created."
				),
				idempotency_key=existing_refund.idempotency_key,
			)
			state.response = (
				f"Your existing refund of ₹{existing_refund.amount:.0f} "
				f"is {existing_refund.status.lower()}. No duplicate "
				"refund has been created."
			)

		self.cases.save(state)

		if (
			state.settlement is not None
			and state.settlement.status == SettlementStatus.AUTO_APPROVED
			and state.settlement.total_refund > 0
			and existing_refund is None
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
		existing_refund = self.refund_workflow.settlement.ledger.get_refund(
			state.order_id
		)
		if (
			existing_refund is not None
			and existing_refund.status in {"PROCESSING", "REFUNDED"}
		):
			return existing_refund

		settlement = state.settlement
		if (
			settlement is None
			or settlement.status not in {
				SettlementStatus.AUTO_APPROVED,
				SettlementStatus.PROPOSED,
			}
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
