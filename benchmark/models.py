from typing import Any

from pydantic import BaseModel


class ExpectedGrievance(BaseModel):
	type: str
	claim: str


class ExpectedJudgement(BaseModel):
	grievance_type: str
	verdict: str


class ExpectedSettlement(BaseModel):
	decision: str
	refund_amount: float
	refund_status: str | None = None
	idempotency_key: str | None = None


class ExistingRefund(BaseModel):
	status: str
	amount: float
	idempotency_key: str


class Preconditions(BaseModel):
	existing_refund: ExistingRefund | None = None


class ExpectedOutcome(BaseModel):
	intent: str
	grievances: list[ExpectedGrievance]
	judgements: list[ExpectedJudgement]
	settlement: ExpectedSettlement
	escalation: bool
	answer: str


class GoldenCase(BaseModel):
	case_id: str
	category: str
	order_id: str
	customer_message: str
	expected: ExpectedOutcome
	preconditions: Preconditions | None = None


class ActualGrievance(BaseModel):
	type: str


class ActualJudgement(BaseModel):
	grievance_type: str
	verdict: str


class ActualSettlement(BaseModel):
	decision: str | None
	refund_amount: float
	refund_status: str | None = None
	idempotency_key: str | None = None


class ActualOutcome(BaseModel):
	intent: str
	grievances: list[ActualGrievance]
	judgements: list[ActualJudgement]
	settlement: ActualSettlement
	escalation: bool


class FieldComparison(BaseModel):
	field: str
	expected: Any
	actual: Any
	passed: bool


class CasePerformance(BaseModel):
	latency_ms: float | None = None
	llm_call_count: int = 0
	judge_call_count: int = 0
	retry_count: int = 0


class BenchmarkResult(BaseModel):
	case_id: str
	passed: bool
	comparisons: list[FieldComparison]
	performance: CasePerformance | None = None
