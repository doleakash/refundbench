from __future__ import annotations

import json

from openai import NotFoundError, PermissionDeniedError

from app.domain.models import Grievance
from app.infrastructure.llm_model import LLMModel


class GrievanceExtractor:

	def __init__(self):
		self.llm = LLMModel()
		self.model = self.llm.model

	def extract(
		self,
		customer_message: str,
		case_id: str | None = None,
		order_id: str | None = None,
	) -> list[Grievance]:

		prompt = f"""
You are a customer-support grievance extraction system.

Extract every distinct grievance from this customer message.

Customer message:
{customer_message}

Supported grievance types:
- LATE_DELIVERY
- MISSING_ITEMS
- LEAKED_ITEM
- FOOD_QUALITY
- WRONG_ITEM

Rules:
- Extract every distinct grievance.
- Do not decide whether the claim is true.
- Do not calculate refunds.
- Do not resolve the case.
- Keep each claim specific.

Return JSON only:

[
    {{
        "grievance_id": "G1",
        "type": "LATE_DELIVERY",
        "claim": "The order was delivered late"
    }}
]
"""

		try:
			response = self.llm.complete(
				[
					{
						"role": "system",
						"content": "Extract customer grievances into structured JSON.",
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

		data = json.loads(content)

		return [
			Grievance.model_validate(item)
			for item in data
		]
