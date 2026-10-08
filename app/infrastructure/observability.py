import logging

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class TraceEvent(BaseModel):
    event_type: str
    message: str


class TimingEvent(BaseModel):
    case_id: str | None
    order_id: str | None
    stage: str
    duration_ms: float
    success: bool
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class AgentTrace(BaseModel):
    case_id: str
    customer_message: str
    events: list[TraceEvent] = Field(default_factory=list)
    timing_events: list[TimingEvent] = Field(default_factory=list)
    actions: list[str]
    observations: list[str]
    tool_errors: list[str]
    response: str | None = None


def log_timing_event(event: TimingEvent) -> None:
    logger.info("agent_timing %s", event.model_dump_json())