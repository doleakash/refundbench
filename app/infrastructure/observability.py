import logging
from contextlib import contextmanager
from threading import Lock
from typing import Iterator

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)
_active_timing_captures: dict[str, list[list["TimingEvent"]]] = {}
_timing_capture_lock = Lock()


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
    provider: str | None = None
    endpoint: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class AgentTrace(BaseModel):
    case_id: str
    customer_message: str
    events: list[TraceEvent] = Field(default_factory=list)
    timing_events: list[TimingEvent] = Field(default_factory=list)
    tool_retry_count: dict[str, int] = Field(default_factory=dict)
    llm_retry_count: int | None = None
    actions: list[str]
    observations: list[str]
    tool_errors: list[str]
    response: str | None = None
    outcome: str | None = None


@contextmanager
def capture_timing_events(case_id: str) -> Iterator[list[TimingEvent]]:
    events: list[TimingEvent] = []
    with _timing_capture_lock:
        _active_timing_captures.setdefault(case_id, []).append(events)
    try:
        yield events
    finally:
        with _timing_capture_lock:
            captures = _active_timing_captures.get(case_id, [])
            if events in captures:
                captures.remove(events)
            if not captures:
                _active_timing_captures.pop(case_id, None)


def captured_timing_events(case_id: str) -> list[TimingEvent] | None:
    with _timing_capture_lock:
        captures = _active_timing_captures.get(case_id)
        return list(captures[-1]) if captures else None


def log_timing_event(event: TimingEvent) -> None:
    if event.case_id is not None:
        with _timing_capture_lock:
            for capture in _active_timing_captures.get(event.case_id, []):
                capture.append(event)
    logger.info("agent_timing %s", event.model_dump_json())
