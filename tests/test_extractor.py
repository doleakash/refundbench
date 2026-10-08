import json
from types import SimpleNamespace

import pytest

from app.agent.orchestrator import run_agent
from app.agent.state import Action
from app.domain.models import CustomerIntent, GrievanceType
from app.evaluation.extractor import (
	GrievanceExtractor,
	RawClaim,
	normalize_claims,
)
from app.infrastructure.observability import TimingEvent


def test_extractor_keeps_intent_separate_from_multiple_grievances(
	monkeypatch,
):
	responses = iter([
		{
			"intent": "REFUND_REQUEST",
			"claims": [
				{
					"raw_claim": "My order was delivered late",
					"type": "LATE_DELIVERY",
					"ambiguous": False,
				},
			],
		},
		{
			"intent": "REFUND_REQUEST",
			"claims": [
				{
					"raw_claim": "My order was late",
					"type": "LATE_DELIVERY",
					"ambiguous": False,
				},
				{
					"raw_claim": "Two items were missing",
					"type": "MISSING_ITEMS",
					"ambiguous": False,
				},
			],
		},
	])

	class FakeLLM:
		model = "test-model"

		def complete(self, messages, case_id=None, order_id=None):
			content = json.dumps(next(responses))
			return SimpleNamespace(
				choices=[
					SimpleNamespace(
						message=SimpleNamespace(content=content)
					)
				]
			), TimingEvent(
				case_id=case_id,
				order_id=order_id,
				stage="LLM_EXTRACT_GRIEVANCES",
				duration_ms=0.0,
				success=True,
				model=self.model,
			)

	monkeypatch.setattr("app.evaluation.extractor.LLMModel", FakeLLM)
	extractor = GrievanceExtractor()

	single = extractor.extract(
		"My order was delivered late. I want a refund."
	)
	assert extractor.intent is CustomerIntent.REFUND_REQUEST
	assert [grievance.type for grievance in single] == [
		GrievanceType.LATE_DELIVERY,
	]
	assert single[0].raw_claims == ["My order was delivered late"]

	multiple = extractor.extract(
		"My order was late and two items were missing. I want a refund."
	)
	assert extractor.intent is CustomerIntent.REFUND_REQUEST
	assert [grievance.type for grievance in multiple] == [
		GrievanceType.LATE_DELIVERY,
		GrievanceType.MISSING_ITEMS,
	]
	assert [claim.raw_claim for claim in extractor.raw_claims] == [
		"My order was late",
		"Two items were missing",
	]


def test_extractor_prompt_excludes_resolution_requests_from_grievances(
	monkeypatch,
):
	class FakeLLM:
		model = "test-model"
		prompt = ""

		def complete(self, messages, case_id=None, order_id=None):
			self.prompt = messages[-1]["content"]
			return SimpleNamespace(
				choices=[
					SimpleNamespace(
						message=SimpleNamespace(content=json.dumps({
							"intent": "REFUND_REQUEST",
							"claims": [{
								"raw_claim": "The raita was missing",
								"type": "MISSING_ITEMS",
								"ambiguous": False,
							}],
						}))
					)
				]
			), TimingEvent(
				case_id=case_id,
				order_id=order_id,
				stage="LLM_EXTRACT_GRIEVANCES",
				duration_ms=0.0,
				success=True,
				model=self.model,
			)

	monkeypatch.setattr("app.evaluation.extractor.LLMModel", FakeLLM)
	extractor = GrievanceExtractor()
	grievances = extractor.extract(
		"The raita was missing, so please refund its price."
	)

	assert extractor.intent is CustomerIntent.REFUND_REQUEST
	assert [grievance.type for grievance in grievances] == [
		GrievanceType.MISSING_ITEMS,
	]
	assert "they are not grievances" in extractor.llm.prompt
	assert "never requested remedies" in extractor.llm.prompt


def test_extractor_prompt_recognizes_referenced_grievance_without_restatement(
	monkeypatch,
):
	class FakeLLM:
		model = "test-model"
		prompt = ""

		def complete(self, messages, case_id=None, order_id=None):
			self.prompt = messages[-1]["content"]
			return SimpleNamespace(
				choices=[
					SimpleNamespace(
						message=SimpleNamespace(content=json.dumps({
							"intent": "REFUND_REQUEST",
							"claims": [{
								"raw_claim": (
									"the same late-delivery refund request"
								),
								"type": "LATE_DELIVERY",
								"ambiguous": False,
							}],
						}))
					)
				]
			), TimingEvent(
				case_id=case_id,
				order_id=order_id,
				stage="LLM_EXTRACT_GRIEVANCES",
				duration_ms=0.0,
				success=True,
				model=self.model,
			)

	monkeypatch.setattr("app.evaluation.extractor.LLMModel", FakeLLM)
	extractor = GrievanceExtractor()
	grievances = extractor.extract(
		"I am submitting the same late-delivery refund request again. "
		"Please do not create a duplicate refund."
	)

	assert extractor.intent is CustomerIntent.REFUND_REQUEST
	assert [grievance.type for grievance in grievances] == [
		GrievanceType.LATE_DELIVERY,
	]
	assert grievances[0].raw_claims == [
		"the same late-delivery refund request",
	]
	assert "References to a previous or repeated request" in (
		extractor.llm.prompt
	)
	assert "Do not infer an issue when the message does not identify one." in (
		extractor.llm.prompt
	)


def test_extractor_preserves_reported_issue_challenged_by_app_evidence(
	monkeypatch,
):
	class FakeLLM:
		model = "test-model"
		prompt = ""

		def complete(self, messages, case_id=None, order_id=None):
			self.prompt = messages[-1]["content"]
			return SimpleNamespace(
				choices=[
					SimpleNamespace(
						message=SimpleNamespace(content=json.dumps({
							"intent": "DELIVERY_SUPPORT",
							"claims": [{
								"raw_claim": "The delivery may have been late",
								"type": "LATE_DELIVERY",
								"ambiguous": False,
							}],
						}))
					)
				]
			), TimingEvent(
				case_id=case_id,
				order_id=order_id,
				stage="LLM_EXTRACT_GRIEVANCES",
				duration_ms=0.0,
				success=True,
				model=self.model,
			)

	monkeypatch.setattr("app.evaluation.extractor.LLMModel", FakeLLM)
	extractor = GrievanceExtractor()
	grievances = extractor.extract(
		"I thought the delivery was late, but the app says the driver "
		"arrived before the promised time. Can you check?",
	)

	assert extractor.intent is CustomerIntent.DELIVERY_SUPPORT
	assert [grievance.type for grievance in grievances] == [
		GrievanceType.LATE_DELIVERY,
	]
	assert grievances[0].claim == "The delivery may have been late"
	assert "judges, not extraction, determine" in extractor.llm.prompt


def test_normalize_claims_attaches_related_ambiguity_and_preserves_sources():
	raw_claims = [
		RawClaim(
			raw_claim="My order was delivered 85 minutes late",
			type=GrievanceType.LATE_DELIVERY,
		),
		RawClaim(
			raw_claim="Two biryanis were missing",
			type=GrievanceType.MISSING_ITEMS,
		),
		RawClaim(
			raw_claim="The raita container leaked everywhere",
			type=GrievanceType.LEAKED_ITEM,
		),
		RawClaim(
			raw_claim="Food is burned",
			type=GrievanceType.FOOD_QUALITY,
		),
		RawClaim(
			raw_claim="Food is red colour",
			type=GrievanceType.FOOD_QUALITY,
			ambiguous=True,
		),
		RawClaim(
			raw_claim="It rotten",
			type=GrievanceType.FOOD_QUALITY,
			ambiguous=True,
		),
		RawClaim(
			raw_claim="How can I eat this",
			type=GrievanceType.FOOD_QUALITY,
			ambiguous=True,
		),
		RawClaim(
			raw_claim="Do I get poisoned with this",
			type=GrievanceType.FOOD_QUALITY,
			ambiguous=True,
		),
		RawClaim(
			raw_claim="Also it smells foul",
			type=GrievanceType.FOOD_QUALITY,
			ambiguous=True,
		),
		RawClaim(
			raw_claim="I think this is chicken instead of mutton",
			type=GrievanceType.WRONG_ITEM,
		),
	]
	grievances = normalize_claims(raw_claims)

	assert len(raw_claims) == 10
	assert [grievance.type for grievance in grievances] == [
		GrievanceType.LATE_DELIVERY,
		GrievanceType.MISSING_ITEMS,
		GrievanceType.LEAKED_ITEM,
		GrievanceType.FOOD_QUALITY,
		GrievanceType.WRONG_ITEM,
	]
	assert grievances[3].raw_claims == [
		claim.raw_claim for claim in raw_claims[3:9]
	]
	assert grievances[3].claim == "; ".join(
		claim.raw_claim for claim in raw_claims[3:9]
	)
	assert len(grievances) == 5

	ambiguous_without_clear_association = normalize_claims([
		RawClaim(
			raw_claim="The food was burned",
			type=GrievanceType.FOOD_QUALITY,
		),
		RawClaim(
			raw_claim="There is something unclear",
			type=GrievanceType.FOOD_QUALITY,
			ambiguous=True,
		),
	])
	assert len(ambiguous_without_clear_association) == 1
	assert ambiguous_without_clear_association[0].raw_claims == [
		"The food was burned",
		"There is something unclear",
	]


def test_ambiguous_claim_attaches_only_to_an_existing_same_type():
	grievances = normalize_claims([
		RawClaim(
			raw_claim="The meat was the wrong cut",
			type=GrievanceType.WRONG_ITEM,
		),
		RawClaim(
			raw_claim="I think it is mutton",
			type=GrievanceType.WRONG_ITEM,
			ambiguous=True,
		),
		RawClaim(
			raw_claim="There is something unclear",
			type=GrievanceType.FOOD_QUALITY,
			ambiguous=True,
		),
	])

	assert [grievance.type for grievance in grievances] == [
		GrievanceType.WRONG_ITEM,
		GrievanceType.FOOD_QUALITY,
	]
	assert grievances[0].raw_claims == [
		"The meat was the wrong cut",
		"I think it is mutton",
	]
	assert grievances[1].raw_claims == ["There is something unclear"]


def test_normalize_claims_merges_non_ambiguous_canonical_duplicates():
	grievances = normalize_claims([
		RawClaim(
			raw_claim="The order was late!",
			type=GrievanceType.LATE_DELIVERY,
		),
		RawClaim(
			raw_claim="THE ORDER WAS LATE",
			type=GrievanceType.LATE_DELIVERY,
		),
	])

	assert len(grievances) == 1
	assert grievances[0].claim == "The order was late!"
	assert grievances[0].raw_claims == [
		"The order was late!",
		"THE ORDER WAS LATE",
	]


@pytest.mark.parametrize("content", [
	"{not json",
	'{"intent": "NOT_AN_INTENT", "claims": []}',
	(
		'{"intent": "REFUND_REQUEST", "claims": ['
		'{"raw_claim": "Late", "type": "NOT_A_GRIEVANCE"}]}'
	),
	(
		'{"intent": "REFUND_REQUEST", "claims": ['
		'{"type": "LATE_DELIVERY"}]}'
	),
])
def test_invalid_extraction_output_escalates_without_downstream_work(
	content,
	monkeypatch,
):
	class FakeLLM:
		model = "test-model"

		def complete(self, messages, case_id=None, order_id=None):
			return SimpleNamespace(
				choices=[
					SimpleNamespace(
						message=SimpleNamespace(content=content)
					)
				]
			), TimingEvent(
				case_id=case_id,
				order_id=order_id,
				stage="LLM_EXTRACT_GRIEVANCES",
				duration_ms=0.0,
				success=True,
				model=self.model,
			)

	monkeypatch.setattr("app.evaluation.extractor.LLMModel", FakeLLM)

	state = run_agent(
		customer_message="My order was late.",
		order_id="ORD-123",
		case_id="CASE-INVALID-EXTRACTION",
	)

	assert state.actions == [
		Action.GET_ORDER.value,
		Action.GET_DELIVERY.value,
		Action.EXTRACT_GRIEVANCES.value,
	]
	assert state.escalation_reason is None
	assert "Tool failure requires human escalation." in state.observations
	assert state.evidence == {}
	assert state.judgments == {}
	assert state.policy_decisions == {}
	assert state.settlement is None
	assert any(
		event.stage == "LLM_EXTRACT_GRIEVANCES" and not event.success
		for event in state.timing_events
	)
	assert any(
		"malformed intent and claim extraction output" in error
		for error in state.tool_errors
	)