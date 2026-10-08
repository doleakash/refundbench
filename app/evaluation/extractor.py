from __future__ import annotations

import re

from openai import NotFoundError, PermissionDeniedError
from pydantic import BaseModel, ValidationError

from app.domain.models import (
	CustomerIntent,
	Grievance,
	GrievanceType,
)
from app.infrastructure.observability import TimingEvent
from app.infrastructure.llm_model import LLMModel


class RawClaim(BaseModel):
	raw_claim: str
	type: GrievanceType
	ambiguous: bool = False


class ExtractionResult(BaseModel):
	intent: CustomerIntent
	claims: list[RawClaim]


def normalize_claims(raw_claims: list[RawClaim]) -> list[Grievance]:
	normalized: list[
		tuple[GrievanceType, list[str], list[str]]
	] = []
	claim_indexes: dict[tuple[GrievanceType, str], int] = {}
	type_indexes: dict[GrievanceType, int] = {}
	food_quality_index: int | None = None
	food_quality_canonical_claims: set[str] = set()

	for raw_claim in raw_claims:
		if not raw_claim.raw_claim.strip():
			continue

		cleaned_claim = raw_claim.raw_claim.strip()
		canonical_claim = re.sub(
			r"[\W_]+",
			" ",
			cleaned_claim.casefold(),
		).strip()
		if raw_claim.type is GrievanceType.FOOD_QUALITY:
			if raw_claim.ambiguous:
				if food_quality_index is not None:
					grievance = normalized[food_quality_index]
					grievance[2].append(cleaned_claim)
					if canonical_claim not in food_quality_canonical_claims:
						food_quality_canonical_claims.add(canonical_claim)
						grievance[1].append(cleaned_claim)
					continue
				food_quality_index = len(normalized)
				type_indexes.setdefault(raw_claim.type, food_quality_index)
				normalized.append((
					raw_claim.type,
					[cleaned_claim],
					[cleaned_claim],
				))
				continue
			if food_quality_index is None:
				food_quality_index = len(normalized)
				type_indexes.setdefault(raw_claim.type, food_quality_index)
				food_quality_canonical_claims.add(canonical_claim)
				normalized.append((
					raw_claim.type,
					[cleaned_claim],
					[cleaned_claim],
				))
				continue
			grievance = normalized[food_quality_index]
			grievance[2].append(cleaned_claim)
			if canonical_claim not in food_quality_canonical_claims:
				food_quality_canonical_claims.add(canonical_claim)
				grievance[1].append(cleaned_claim)
			continue

		if raw_claim.ambiguous:
			existing_index = type_indexes.get(raw_claim.type)
			if existing_index is not None:
				grievance = normalized[existing_index]
				grievance[2].append(cleaned_claim)
				if canonical_claim not in {
					re.sub(r"[\W_]+", " ", claim.casefold()).strip()
					for claim in grievance[1]
				}:
					grievance[1].append(cleaned_claim)
				continue
			type_indexes[raw_claim.type] = len(normalized)
			normalized.append((
				raw_claim.type,
				[cleaned_claim],
				[cleaned_claim],
			))
			continue

		key = (raw_claim.type, canonical_claim)
		existing_index = claim_indexes.get(key)
		if existing_index is not None:
			normalized[existing_index][2].append(cleaned_claim)
			continue
		claim_indexes[key] = len(normalized)
		type_indexes.setdefault(raw_claim.type, len(normalized))
		normalized.append((
			raw_claim.type,
			[cleaned_claim],
			[cleaned_claim],
		))

	return [
		Grievance(
			grievance_id=f"G{index}",
			type=grievance_type,
			claim="; ".join(normalized_claims),
			raw_claims=source_claims,
		)
		for index, (grievance_type, normalized_claims, source_claims)
		in enumerate(normalized, start=1)
	]


class GrievanceExtractor:

	def __init__(self):
		self.llm = LLMModel()
		self.model = self.llm.model
		self.intent = CustomerIntent.GENERAL_SUPPORT
		self.raw_claims: list[RawClaim] = []
		self.timing_event: TimingEvent | None = None

	def extract(
		self,
		customer_message: str,
		case_id: str | None = None,
		order_id: str | None = None,
	) -> list[Grievance]:

		prompt = f"""
You are a customer-support intent classification and raw-claim extraction system.

Classify the customer's intent separately from the order problems they report.
Intent means what the customer is trying to achieve; grievances mean what went
wrong. Extract raw claims, assigning a supported grievance type when possible.
Omit irrelevant statements. Assign an issue that is difficult to classify to
its closest supported type and set ambiguous to true; ambiguous claims must
remain separate. Preserve distinct business issues as separate claims.
Combine multiple descriptions of food quality into separate raw claims; the
application will normalize them.

A grievance must describe an actual problem or failure with the order.
Requests for a refund, compensation, replacement, cancellation, or another
resolution express customer intent; they are not grievances and must not be
included in claims. When a message reports a problem and asks for a resolution,
extract the problem as a claim and classify the requested resolution as intent.
Extract a reported problem even when the customer also cites information that
may contradict it; judges, not extraction, determine whether the problem is
supported by evidence.
References to a previous or repeated request can identify its underlying
grievance even when the customer does not restate the original problem. If the
message names the issue type (for example, "the same late-delivery refund
request"), extract that issue as a claim and preserve the customer's wording.
Do not infer an issue when the message does not identify one.

Customer message:
{customer_message}

Supported intents:
- REFUND_REQUEST
- REFUND_STATUS
- ORDER_SUPPORT
- DELIVERY_SUPPORT
- GENERAL_SUPPORT

Supported grievance types:
- LATE_DELIVERY
- MISSING_ITEMS
- LEAKED_ITEM
- FOOD_QUALITY
- WRONG_ITEM

Rules:
- Preserve the customer's wording in raw_claim.
- Do not omit a refund request when grievances are also present.
- Extract only order problems or failures as claims, never requested remedies.
- Do not decide whether the claim is true.
- Do not calculate refunds.
- Do not resolve the case.
- Keep each claim specific.

Return JSON only:

{{
  "intent": "REFUND_REQUEST",
  "claims": [
    {{
        "raw_claim": "The order was delivered late",
        "type": "LATE_DELIVERY",
        "ambiguous": false
    }}
  ]
}}
"""

		try:
			response, timing_event = self.llm.complete(
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
			self.timing_event = timing_event
		except (NotFoundError, PermissionDeniedError) as error:
			raise ValueError(
				f"Configured OPEN_AI_MODEL '{self.model}' is unavailable "
				"to this OpenAI account."
			) from error

		content = response.choices[0].message.content
		if content is None:
			raise RuntimeError(
				"LLM returned malformed intent and claim extraction output."
			)
		try:
			result = ExtractionResult.model_validate_json(content)
		except (ValidationError, TypeError) as error:
			raise RuntimeError(
				"LLM returned malformed intent and claim extraction output."
			) from error

		self.intent = result.intent
		self.raw_claims = result.claims
		return normalize_claims(self.raw_claims)
