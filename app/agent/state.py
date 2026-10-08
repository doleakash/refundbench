from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field

from app.domain.models import CustomerIntent, Delivery, Grievance, Order
from app.infrastructure.observability import TimingEvent


class Action(str, Enum):
    GET_ORDER = "GET_ORDER"
    GET_DELIVERY = "GET_DELIVERY"
    EXTRACT_GRIEVANCES = "EXTRACT_GRIEVANCES"
    GET_EVIDENCE = "GET_EVIDENCE"
    RUN_JUDGES = "RUN_JUDGES"
    BUILD_CONSENSUS = "BUILD_CONSENSUS"
    APPLY_POLICY = "APPLY_POLICY"
    SETTLE = "SETTLE"
    ESCALATE = "ESCALATE"
    STOP = "STOP"


class AgentState(BaseModel):
    # Request identity
    customer_message: str
    order_id: str
    case_id: str
    intent: CustomerIntent = CustomerIntent.GENERAL_SUPPORT

    # Agent execution state
    actions: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    tool_errors: list[str] = Field(default_factory=list)
    tool_retry_count: dict[str, int] = Field(default_factory=dict)
    timing_events: list[TimingEvent] = Field(default_factory=list)

    # Retrieved business data
    order: Order | None = None
    delivery: Delivery | None = None

    # Resolution pipeline state
    grievances: list[Grievance] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)
    judgments: dict = Field(default_factory=dict)
    consensus: dict = Field(default_factory=dict)
    policy_decisions: dict = Field(default_factory=dict)

    # Final resolution
    settlement: object | None = None
    response: str | None = None

    # Failure / escalation state
    escalation_reason: str | None = None


class AgentDecision(BaseModel):
    action: Action


class AgentModel(ABC):

    @abstractmethod
    def decide(self, state: AgentState) -> AgentDecision:
        raise NotImplementedError