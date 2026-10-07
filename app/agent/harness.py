from app.agent.state import Action, AgentDecision, AgentState


class AgentHarness:
	ALLOWED_ACTIONS = {
		Action.GET_ORDER,
		Action.GET_DELIVERY,
		Action.EXTRACT_GRIEVANCES,
		Action.GET_EVIDENCE,
		Action.RUN_JUDGES,
		Action.BUILD_CONSENSUS,
		Action.APPLY_POLICY,
		Action.SETTLE,
		Action.ESCALATE,
		Action.STOP,
	}

	def validate(
		self,
		state: AgentState,
		decision: AgentDecision,
	) -> Action:

		action = decision.action

		# ---------------------------------------------------------
		# Basic action validation
		# ---------------------------------------------------------

		if action not in self.ALLOWED_ACTIONS:
			raise ValueError(
				f"Action not allowed: {action}"
			)

		# ---------------------------------------------------------
		# Data dependencies
		# ---------------------------------------------------------

		if action is Action.GET_ORDER:
			return action

		if action is Action.GET_DELIVERY:
			if state.order is None:
				raise ValueError(
					"Cannot get delivery before retrieving the order"
				)

		if action is Action.EXTRACT_GRIEVANCES:
			if not state.customer_message:
				raise ValueError(
					"Cannot extract grievances without a customer message"
				)

			if state.order is None:
				raise ValueError(
					"Cannot extract grievances before retrieving the order"
				)

			if state.delivery is None:
				raise ValueError(
					"Cannot extract grievances before retrieving the delivery"
				)

		if action is Action.GET_EVIDENCE:
			if state.order is None:
				raise ValueError(
					"Cannot get evidence before retrieving the order"
				)

			if state.delivery is None:
				raise ValueError(
					"Cannot get evidence before retrieving the delivery"
				)

			if not state.grievances:
				raise ValueError(
					"Cannot get evidence before extracting grievances"
				)

		# ---------------------------------------------------------
		# Evaluation dependencies
		# ---------------------------------------------------------

		if action is Action.RUN_JUDGES:
			if state.order is None:
				raise ValueError(
					"Cannot run judges before retrieving the order"
				)

			if state.delivery is None:
				raise ValueError(
					"Cannot run judges before retrieving the delivery"
				)

			if not state.grievances:
				raise ValueError(
					"Cannot run judges before extracting grievances"
				)

			if len(state.evidence) != len(state.grievances):
				raise ValueError(
					"Cannot run judges before evidence is available "
					"for all grievances"
				)

		if action is Action.BUILD_CONSENSUS:
			if len(state.judgments) != len(state.grievances):
				raise ValueError(
					"Cannot build consensus before judgments "
					"are available for all grievances"
				)

		# ---------------------------------------------------------
		# Policy / settlement dependencies
		# ---------------------------------------------------------

		if action is Action.APPLY_POLICY:
			if len(state.consensus) != len(state.grievances):
				raise ValueError(
					"Cannot apply policy before consensus "
					"is available for all grievances"
				)

		if action is Action.SETTLE:
			if len(state.policy_decisions) != len(state.grievances):
				raise ValueError(
					"Cannot settle before policy decisions "
					"are available for all grievances"
				)

		# ---------------------------------------------------------
		# Terminal actions
		# ---------------------------------------------------------

		if action is Action.STOP:
			if state.settlement is None:
				raise ValueError(
					"Cannot stop before settlement is completed"
				)

		return action
