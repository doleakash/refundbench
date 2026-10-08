from pydantic import BaseModel, Field

from app.infrastructure.observability import TimingEvent


class CaseRequest(BaseModel):
	order_id: str = Field(min_length=1)
	customer_message: str = Field(min_length=1)


class SettlementResult(BaseModel):
	status: str
	amount: float
	reason: str


class EvidenceResult(BaseModel):
	source: str
	facts: list[str]
	requires_customer_input: bool


class JudgmentResult(BaseModel):
	verdict: str
	reason: str
	confidence: float


class PolicyDecisionResult(BaseModel):
	action: str
	refund_amount: float
	reason: str


class GrievanceResult(BaseModel):
	grievance_id: str
	type: str
	claim: str
	raw_claims: list[str] = Field(default_factory=list)
	evidence: EvidenceResult | None = None
	judgments: list[JudgmentResult] = Field(default_factory=list)
	consensus: str | None = None
	policy_decision: PolicyDecisionResult | None = None


class CaseResult(BaseModel):
	case_id: str
	order_id: str
	intent: str
	status: str
	response: str | None
	settlement: SettlementResult | None
	grievances: list[GrievanceResult]
	actions: list[str]
	observations: list[str]
	tool_errors: list[str]
	escalation_reason: str | None
	timing_events: list[TimingEvent]


class RefundAcceptanceResult(BaseModel):
	case_id: str
	order_id: str
	status: str
	amount: float
	idempotency_key: str
