from __future__ import annotations

from openai import NotFoundError, PermissionDeniedError
from pydantic import BaseModel

from app.domain.models import Evidence, Grievance
from app.infrastructure.observability import TimingEvent
from app.infrastructure.llm_model import LLMModel


class Judgment(BaseModel):
	grievance_id: str
	verdict: str
	reason: str
	confidence: float


class Judge:
	def __init__(self):
		self.llm = LLMModel()
		self.model = self.llm.model

	def judge(
		self,
		grievance: Grievance,
		evidence: Evidence,
		case_id: str | None = None,
		order_id: str | None = None,
	) -> tuple[Judgment, TimingEvent]:

		prompt = f"""
You are a refund claim judge.

Evaluate whether the underlying grievance is supported by the
available evidence.

Customer grievance:
{grievance.claim}

Grievance type:
{grievance.type.value}

Evidence source:
{evidence.source}

Evidence:
{evidence.facts}

Important rules:
- Judge only based on the provided evidence.
- Never invent or assume evidence.
- Do not calculate a refund.
- Do not make the final settlement decision.
- If the available evidence cannot establish the grievance, return INSUFFICIENT_EVIDENCE.
- Do not reject a grievance solely because a specific numeric detail in the customer's claim is inaccurate.
- Evaluate whether the underlying grievance is supported by the evidence.
- For example, if the customer claims the order was 85 minutes late but the system evidence shows a 70-minute delay, the underlying grievance "late delivery" is still supported. Return UPHELD and mention the discrepancy in the reason.
- Do not infer an exact item count from an aggregate measurement such as weight unless the evidence explicitly establishes that relationship.
- A weight discrepancy may indicate that some quantity is missing, but it does not by itself prove the exact number of missing items.
- Return REJECTED only when the available evidence contradicts the underlying grievance.

Allowed verdicts:
- UPHELD
- REJECTED
- INSUFFICIENT_EVIDENCE

Return JSON only:

{{
    "grievance_id": "{grievance.grievance_id}",
    "verdict": "UPHELD",
    "reason": "The evidence supports the underlying grievance.",
    "confidence": 0.95
}}
"""

		try:
			response, timing_event = self.llm.complete(
				[
					{
						"role": "system",
						"content": "You are an independent refund claim judge.",
					},
					{
						"role": "user",
						"content": prompt,
					},
				],
				case_id=case_id,
				order_id=order_id,
			)
		except (NotFoundError, PermissionDeniedError) as error:
			raise ValueError(
				f"Configured OPEN_AI_MODEL '{self.model}' is unavailable "
				"to this OpenAI account."
			) from error

		content = response.choices[0].message.content

		return Judgment.model_validate_json(content), timing_event
