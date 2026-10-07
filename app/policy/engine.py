from enum import Enum

from app.domain.models import Evidence, Grievance


class PolicyAction(str, Enum):
	REFUND = "REFUND"
	REQUEST_EVIDENCE = "REQUEST_EVIDENCE"
	ESCALATE = "ESCALATE"
	NO_REFUND = "NO_REFUND"


class PolicyDecision:
	def __init__(
		self,
		grievance_id: str,
		action: PolicyAction,
		refund_amount: float = 0.0,
		reason: str = "",
	):
		self.grievance_id = grievance_id
		self.action = action
		self.refund_amount = refund_amount
		self.reason = reason

	def __repr__(self):
		return (
			f"PolicyDecision("
			f"grievance_id={self.grievance_id}, "
			f"action={self.action}, "
			f"refund_amount={self.refund_amount}, "
			f"reason={self.reason})"
		)


class Policy:
	def decide(
		self,
		grievance: Grievance,
		evidence: Evidence,
		consensus: str,
	) -> PolicyDecision:

		# Not enough evidence to make a decision
		if consensus == "INSUFFICIENT_EVIDENCE":

			if evidence.requires_customer_input:
				return PolicyDecision(
					grievance_id=grievance.grievance_id,
					action=PolicyAction.REQUEST_EVIDENCE,
					reason="Additional customer evidence is required.",
				)

			return PolicyDecision(
				grievance_id=grievance.grievance_id,
				action=PolicyAction.ESCALATE,
				reason="Insufficient evidence to resolve the grievance.",
			)

		# Grievance was rejected
		if consensus == "REJECTED":
			return PolicyDecision(
				grievance_id=grievance.grievance_id,
				action=PolicyAction.NO_REFUND,
				reason="The grievance was not supported by the evidence.",
			)

		# Grievance was upheld
		if consensus == "UPHELD":

			if grievance.type.value == "LATE_DELIVERY":
				return PolicyDecision(
					grievance_id=grievance.grievance_id,
					action=PolicyAction.REFUND,
					refund_amount=100.0,
					reason="Late delivery grievance upheld.",
				)

			if grievance.type.value == "MISSING_ITEMS":

				missing_amount = 0.0

				for fact in evidence.facts:
					if "missing:" not in fact or "unit price:" not in fact:
						continue

					parts = fact.split(",")

					missing_quantity = 0
					unit_price = 0.0

					for part in parts:
						part = part.strip()

						if part.startswith("missing:"):
							missing_quantity = int(
								part.split(":")[1].strip()
							)

						elif part.startswith("unit price:"):
							unit_price = float(
								part.split("₹")[1]
							)

					missing_amount += missing_quantity * unit_price

				if missing_amount <= 0:
					return PolicyDecision(
						grievance_id=grievance.grievance_id,
						action=PolicyAction.ESCALATE,
						refund_amount=0.0,
						reason=(
							"Missing items were upheld, but the "
							"refund amount could not be determined."
						),
					)

				return PolicyDecision(
					grievance_id=grievance.grievance_id,
					action=PolicyAction.REFUND,
					refund_amount=missing_amount,
					reason=(
						"Refund calculated from verified "
						"missing item quantities."
					),
				)

			return PolicyDecision(
				grievance_id=grievance.grievance_id,
				action=PolicyAction.ESCALATE,
				reason="No refund policy is defined for this grievance type.",
			)

		# Unknown consensus
		return PolicyDecision(
			grievance_id=grievance.grievance_id,
			action=PolicyAction.ESCALATE,
			reason="Unknown consensus result.",
		)
