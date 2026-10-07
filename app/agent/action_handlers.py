from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Callable

from app.agent.state import Action, AgentState
from app.evaluation.consensus import Consensus
from app.evaluation.evidence import get_evidence
from app.evaluation.extractor import GrievanceExtractor
from app.evaluation.judge import Judge
from app.infrastructure.observability import TimingEvent, log_timing_event
from app.infrastructure.tools import get_delivery, get_order
from app.policy.engine import Policy
from app.response.generator import generate_grievance_response
from app.settlement.settlement import Settlement
from config.settings import get_settings

MAX_PARALLEL_JUDGE_CALLS = get_settings().max_parallel_judge_calls
TimingRecorder = Callable[
	[AgentState, str, float, bool, str | None],
	None,
]


class AgentActionHandlers:
	def __init__(
		self,
		settlement: Settlement,
		record_timing: TimingRecorder,
	):
		self.settlement = settlement
		self.record_timing = record_timing

	def execute(self, state: AgentState, action: Action) -> None:
		handlers = {
			Action.GET_ORDER: self._get_order,
			Action.GET_DELIVERY: self._get_delivery,
			Action.EXTRACT_GRIEVANCES: self._extract_grievances,
			Action.GET_EVIDENCE: self._get_evidence,
			Action.RUN_JUDGES: self._run_judges,
			Action.BUILD_CONSENSUS: self._build_consensus,
			Action.APPLY_POLICY: self._apply_policy,
			Action.SETTLE: self._settle,
			Action.ESCALATE: self._escalate,
			Action.STOP: self._stop,
		}
		handlers[action](state)

	def _get_order(self, state: AgentState) -> None:
		state.order = get_order(state.order_id)
		if state.order is None:
			state.observations.append("Order not found.")
		else:
			state.observations.append(
				f"Order found with status {state.order.status}."
			)

	def _get_delivery(self, state: AgentState) -> None:
		state.delivery = get_delivery(state.order_id)
		if state.delivery is None:
			state.observations.append("Delivery record not found.")
		else:
			state.observations.append(
				f"Delivery found with status {state.delivery.status}."
			)

	def _extract_grievances(self, state: AgentState) -> None:
		extractor = GrievanceExtractor()
		started_at = perf_counter()
		succeeded = False
		try:
			state.grievances = extractor.extract(
				state.customer_message,
				case_id=state.case_id,
				order_id=state.order_id,
			)
			succeeded = True
		finally:
			self.record_timing(
				state,
				"LLM_EXTRACT_GRIEVANCES",
				started_at,
				succeeded,
				extractor.model,
			)

		state.observations.append(
			f"Extracted {len(state.grievances)} grievances."
		)

	def _get_evidence(self, state: AgentState) -> None:
		if state.order is None:
			raise ValueError("Cannot get evidence before retrieving the order.")
		if state.delivery is None:
			raise ValueError(
				"Cannot get evidence before retrieving the delivery."
			)

		for grievance in state.grievances:
			state.evidence[grievance.grievance_id] = get_evidence(
				grievance=grievance,
				order=state.order,
				delivery=state.delivery,
			)

		state.observations.append(
			f"Collected evidence for {len(state.grievances)} grievances."
		)

	def _run_judges(self, state: AgentState) -> None:
		judge = Judge()

		def run_judge(grievance, evidence, judge_number):
			started_at = perf_counter()
			stage = (
				f"LLM_JUDGE:{grievance.grievance_id}:"
				f"{judge_number}"
			)
			try:
				judgment = judge.judge(
					grievance=grievance,
					evidence=evidence,
					case_id=state.case_id,
					order_id=state.order_id,
				)
			except Exception as error:
				event = TimingEvent(
					case_id=state.case_id,
					order_id=state.order_id,
					stage=stage,
					duration_ms=(perf_counter() - started_at) * 1000,
					success=False,
					model=judge.model,
				)
				return None, event, error

			event = TimingEvent(
				case_id=state.case_id,
				order_id=state.order_id,
				stage=stage,
				duration_ms=(perf_counter() - started_at) * 1000,
				success=True,
				model=judge.model,
			)
			return judgment, event, None

		grievance_tasks = []
		for grievance in state.grievances:
			evidence = state.evidence.get(grievance.grievance_id)
			if evidence is None:
				raise ValueError(
					f"No evidence found for {grievance.grievance_id}."
				)
			grievance_tasks.append((grievance, evidence))

		with ThreadPoolExecutor(
			max_workers=MAX_PARALLEL_JUDGE_CALLS
		) as executor:
			futures = [
				(
					grievance.grievance_id,
					executor.submit(
						run_judge,
						grievance,
						evidence,
						judge_number,
					),
				)
				for grievance, evidence in grievance_tasks
				for judge_number in range(1, 4)
			]
			results = [
				(grievance_id, future.result())
				for grievance_id, future in futures
			]

		judgments_by_grievance = {
			grievance.grievance_id: []
			for grievance, _ in grievance_tasks
		}
		failures_by_grievance = {
			grievance.grievance_id: []
			for grievance, _ in grievance_tasks
		}
		for grievance_id, (judgment, event, error) in results:
			state.timing_events.append(event)
			log_timing_event(event)
			if error is not None:
				failures_by_grievance[grievance_id].append(error)
			else:
				judgments_by_grievance[grievance_id].append(judgment)

		for grievance, _ in grievance_tasks:
			grievance_id = grievance.grievance_id
			failures = failures_by_grievance[grievance_id]
			if failures:
				raise failures[0]

			state.judgments[grievance_id] = (
				judgments_by_grievance[grievance_id]
			)

		state.observations.append(
			f"Ran 3 judges in parallel for {len(state.grievances)} grievances."
		)

	def _build_consensus(self, state: AgentState) -> None:
		consensus_engine = Consensus()
		for grievance in state.grievances:
			judgments = state.judgments.get(grievance.grievance_id)
			if judgments is None:
				raise ValueError(
					f"No judgments found for {grievance.grievance_id}."
				)
			state.consensus[grievance.grievance_id] = (
				consensus_engine.decide(judgments)
			)

		state.observations.append(
			f"Built consensus for {len(state.grievances)} grievances."
		)

	def _apply_policy(self, state: AgentState) -> None:
		policy = Policy()
		for grievance in state.grievances:
			evidence = state.evidence.get(grievance.grievance_id)
			consensus = state.consensus.get(grievance.grievance_id)
			if evidence is None:
				raise ValueError(
					f"No evidence found for {grievance.grievance_id}."
				)
			if consensus is None:
				raise ValueError(
					f"No consensus found for {grievance.grievance_id}."
				)
			state.policy_decisions[grievance.grievance_id] = policy.decide(
				grievance=grievance,
				evidence=evidence,
				consensus=consensus,
			)

		state.observations.append(
			f"Applied policy to {len(state.grievances)} grievances."
		)

	def _settle(self, state: AgentState) -> None:
		decisions = list(state.policy_decisions.values())
		state.settlement = self.settlement.settle(
			decisions=decisions,
			case_id=state.case_id,
			order_id=state.order_id,
		)
		state.response = generate_grievance_response(
			decisions=decisions,
			settlement=state.settlement,
		)
		state.observations.append(
			f"Settlement completed: {state.settlement.status.value}."
		)
		state.observations.append("Customer response generated.")

	@staticmethod
	def _escalate(state: AgentState) -> None:
		state.observations.append("Case escalated to a human.")

	@staticmethod
	def _stop(state: AgentState) -> None:
		state.observations.append("Agent stopped.")
