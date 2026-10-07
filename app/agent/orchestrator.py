import json
import logging
from time import perf_counter
from uuid import uuid4

from app.agent.action_handlers import AgentActionHandlers
from app.agent.state import (
	Action,
	AgentDecision,
	AgentModel,
	AgentState,
)
from app.agent.harness import AgentHarness
from app.failure.classifier import classify_failure
from app.failure.handler import handle_failure
from app.failure.models import FailureAction
from app.infrastructure.observability import (
	AgentTrace,
	TimingEvent,
	TraceEvent,
	log_timing_event,
)
from app.settlement.settlement import Settlement
from app.settlement.refund_provider import MockRefundProvider
from app.settlement.refund_service import RefundService
from config.settings import get_settings

_settings = get_settings()
MAX_ITERATIONS = _settings.max_agent_iterations
MAX_TOOL_RETRIES = _settings.max_tool_retries
logger = logging.getLogger(__name__)


def record_timing(
	state: AgentState,
	stage: str,
	started_at: float,
	success: bool,
	model: str | None = None,
) -> None:
	event = TimingEvent(
		case_id=state.case_id,
		order_id=state.order_id,
		stage=stage,
		duration_ms=(perf_counter() - started_at) * 1000,
		success=success,
		model=model,
	)
	state.timing_events.append(event)
	log_timing_event(event)


def build_trace(state: AgentState) -> AgentTrace:
	events = []

	for action in state.actions:
		events.append(
			TraceEvent(
				event_type="ACTION",
				message=action,
			)
		)

	return AgentTrace(
		case_id=state.case_id,
		customer_message=state.customer_message,
		events=events,
		timing_events=state.timing_events,
		actions=state.actions,
		observations=state.observations,
		tool_errors=state.tool_errors,
		response=state.response,
	)


def _execute_action(
	state: AgentState,
	action: Action,
	handlers: AgentActionHandlers,
) -> FailureAction | None:
	started_at = perf_counter()
	succeeded = False
	state.actions.append(action.value)

	try:
		handlers.execute(state, action)
		succeeded = True
		return None
	except Exception as error:
		state.tool_errors.append(
			f"{action.value}: "
			f"{type(error).__name__}: {error}"
		)
		failure_type = classify_failure(error)
		failure_action = handle_failure(failure_type)
		state.observations.append(
			f"Tool failure: "
			f"{failure_type.value} → "
			f"{failure_action.value}"
		)
		return failure_action
	finally:
		record_timing(
			state,
			action.value,
			started_at,
			succeeded,
		)

def _get_deterministic_next_action(
	state: AgentState,
	completed_action: Action,
) -> Action | None:
	"""
	Returns the next action in the fixed workflow, or None
	when the action is terminal.
	"""

	if completed_action is Action.GET_ORDER:

		if state.order is None:
			state.escalation_reason = (
				"Order could not be found."
			)
			return Action.ESCALATE

		return Action.GET_DELIVERY

	if completed_action is Action.GET_DELIVERY:

		if state.delivery is None:
			state.escalation_reason = (
				"Delivery record could not be found."
			)
			return Action.ESCALATE

		return Action.EXTRACT_GRIEVANCES

	transitions = {
		Action.EXTRACT_GRIEVANCES: Action.GET_EVIDENCE,
		Action.GET_EVIDENCE: Action.RUN_JUDGES,
		Action.RUN_JUDGES: Action.BUILD_CONSENSUS,
		Action.BUILD_CONSENSUS: Action.APPLY_POLICY,
		Action.APPLY_POLICY: Action.SETTLE,
		Action.SETTLE: Action.STOP,
	}

	return transitions.get(completed_action)


def _run_agent(
	customer_message: str,
	order_id: str,
	case_id: str | None = None,
	max_iterations: int = MAX_ITERATIONS,
	model: AgentModel | None = None,
) -> AgentState:
	if max_iterations <= 0:
		raise ValueError(
			"max_iterations must be greater than zero"
		)

	state = AgentState(
		customer_message=customer_message,
		order_id=order_id,
		case_id=case_id
		        or f"CASE-{uuid4().hex[:8].upper()}",
	)

	harness = AgentHarness()

	# One Settlement instance for the entire agent run.
	# This keeps the RefundLedger alive and makes settlement idempotent.
	settlement = Settlement()
	handlers = AgentActionHandlers(
		settlement=settlement,
		record_timing=record_timing,
	)

	refund_service = RefundService(
		ledger=settlement.ledger,
		provider=MockRefundProvider(),
	)

	retry_action: Action | None = None
	deterministic_action: Action | None = Action.GET_ORDER

	for _ in range(max_iterations):

		# ---------------------------------------------------------
		# GET NEXT ACTION
		# ---------------------------------------------------------

		is_retry = retry_action is not None
		if retry_action is not None:

			action = retry_action
			retry_action = None

		elif deterministic_action is not None:

			action = deterministic_action
			deterministic_action = None

		else:
			state.escalation_reason = (
				"No deterministic action is available."
			)
			action = Action.ESCALATE

		try:
			action = harness.validate(
				state,
				AgentDecision(action=action),
			)
		except ValueError as e:
			state.observations.append(
				f"Harness rejected action: {e}"
			)
			state.escalation_reason = (
				"Agent could not safely continue after harness validation."
			)
			action = Action.ESCALATE
			harness.validate(
				state,
				AgentDecision(action=action),
			)

		if (
			not is_retry
			and action.value in state.actions
			and any(
				event.stage == action.value and event.success
				for event in state.timing_events
			)
		):
			state.observations.append(
				f"Repeated action {action.value} was rejected."
			)
			state.escalation_reason = (
				f"Repeated action {action.value} cannot be safely executed."
			)
			action = harness.validate(
				state,
				AgentDecision(action=Action.ESCALATE),
			)

		# ---------------------------------------------------------
		# EXECUTE ACTION
		# ---------------------------------------------------------

		failure_action = _execute_action(
			state=state,
			action=action,
			handlers=handlers,
		)

		# ---------------------------------------------------------
		# HANDLE FAILURE
		# ---------------------------------------------------------

		if failure_action is FailureAction.RETRY:

			retry_count = state.tool_retry_count.get(
				action.value,
				0,
			)

			if retry_count < MAX_TOOL_RETRIES:
				state.tool_retry_count[
					action.value
				] = retry_count + 1

				retry_action = action

				state.observations.append(
					f"Retrying {action.value} "
					f"(attempt {retry_count + 1})."
				)

				continue

			state.observations.append(
				f"Retry limit reached for "
				f"{action.value}."
			)

			state.escalation_reason = (
				f"{action.value} "
				f"failed after retry limit."
			)

			state.actions.append(
				Action.ESCALATE.value
			)

			state.observations.append(
				"Case escalated to a human."
			)

			trace = build_trace(state)

			print(
				trace.model_dump_json(indent=2)
			)

			return state

		if failure_action is FailureAction.ESCALATE:
			state.observations.append(
				"Tool failure requires human escalation."
			)

			trace = build_trace(state)

			print(
				trace.model_dump_json(indent=2)
			)

			return state

		# ---------------------------------------------------------
		# DETERMINISTIC ROUTING
		# ---------------------------------------------------------

		deterministic_action = _get_deterministic_next_action(
			state=state,
			completed_action=action,
		)

		# ---------------------------------------------------------
		# TERMINAL ACTION
		# ---------------------------------------------------------

		if action in {
			Action.STOP,
			Action.ESCALATE,
		}:
			trace = build_trace(state)

			print(
				trace.model_dump_json(indent=2)
			)

			return state

	raise RuntimeError(
		f"Agent exceeded the maximum of "
		f"{max_iterations} iterations"
	)


def run_agent(
	customer_message: str,
	order_id: str,
	case_id: str | None = None,
	max_iterations: int = MAX_ITERATIONS,
	model: AgentModel | None = None,
) -> AgentState:
	"""Run the deterministic workflow; model is retained for call compatibility."""
	started_at = perf_counter()
	resolved_case_id = case_id or f"CASE-{uuid4().hex[:8].upper()}"
	try:
		state = _run_agent(
			customer_message=customer_message,
			order_id=order_id,
			case_id=resolved_case_id,
			max_iterations=max_iterations,
			model=model,
		)
	except Exception:
		event = TimingEvent(
			case_id=resolved_case_id,
			order_id=order_id,
			stage="TOTAL_AGENT",
			duration_ms=(perf_counter() - started_at) * 1000,
			success=False,
		)
		log_timing_event(event)
		logger.info(
			"agent_summary %s",
			json.dumps({
				"case_id": resolved_case_id,
				"order_id": order_id,
				"outcome": "failed",
			}),
		)
		raise

	outcome = (
		"escalated"
		if (
			(state.actions and state.actions[-1] == Action.ESCALATE.value)
			or state.escalation_reason is not None
			or "Tool failure requires human escalation."
			in state.observations
			or (
				state.settlement is not None
				and state.settlement.status.value == "ESCALATE"
			)
		)
		else "completed"
	)
	record_timing(state, "TOTAL_AGENT", started_at, True)
	llm_events = [
		event for event in state.timing_events if event.model is not None
	]
	logger.info(
		"agent_summary %s",
		json.dumps({
			"case_id": state.case_id,
			"order_id": state.order_id,
			"llm_call_count": len(llm_events),
			"judge_call_count": sum(
				event.stage.startswith("LLM_JUDGE:")
				for event in llm_events
			),
			"retry_count": sum(state.tool_retry_count.values()),
			"outcome": outcome,
		}),
	)
	return state


def accept_refund(
	state: AgentState,
	refund_service: RefundService,
):
	if state.settlement is None:
		raise ValueError(
			"Cannot accept refund before settlement"
		)

	if state.settlement.total_refund <= 0:
		raise ValueError(
			"Cannot accept a zero-value refund"
		)

	refund = refund_service.process_refund(
		order_id=state.order_id,
	)

	state.observations.append(
		f"Refund processed for order "
		f"{state.order_id}: {refund.status}."
	)

	return refund
