from time import perf_counter, sleep
from typing import Optional, Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from app.agent.state import (
    AgentDecision,
    AgentModel,
    AgentState,
)
from app.infrastructure.observability import TimingEvent, log_timing_event
from config.settings import get_settings

_LLM_RETRY_BACKOFF = 0.25


class LLMModel(AgentModel):

    def __init__(self):
        settings = get_settings()
        api_key = settings.openai_api_key
        model = settings.open_ai_model

        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured")

        if not model:
            raise ValueError("OPEN_AI_MODEL is not configured")

        self.model = model
        self.max_retries = settings.max_llm_retries

        self.client = OpenAI(
            api_key=api_key,
            timeout=settings.llm_timeout,
            max_retries=0,
        )

    def complete(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        case_id: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> ChatCompletion:
        for attempt in range(self.max_retries + 1):
            started_at = perf_counter()
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
            except (APIConnectionError, APIStatusError) as error:
                retryable = isinstance(error, APIConnectionError) or (
                    isinstance(error, APIStatusError)
                    and (
                        error.status_code == 429
                        or error.status_code >= 500
                    )
                )
                if not retryable or attempt >= self.max_retries:
                    raise

                log_timing_event(
                    TimingEvent(
                        case_id=case_id,
                        order_id=order_id,
                        stage=f"LLM_RETRY:{attempt + 1}",
                        duration_ms=(perf_counter() - started_at) * 1000,
                        success=False,
                        model=self.model,
                    )
                )
                sleep(_LLM_RETRY_BACKOFF * (2**attempt))

        raise RuntimeError("LLM retry loop ended without a response.")

    def decide(self, state: AgentState) -> AgentDecision:

        prompt = f"""
You are the orchestration agent for a customer-support refund system.

Your job is to choose the NEXT action required to process the customer case.

Customer message:
{state.customer_message}

Order ID:
{state.order_id}

CURRENT STATE

Order retrieved:
{state.order is not None}

Delivery retrieved:
{state.delivery is not None}

Grievances:
{state.grievances}

Evidence:
{state.evidence}

Judgments:
{state.judgments}

Consensus:
{state.consensus}

Policy decisions:
{state.policy_decisions}

Settlement:
{state.settlement!r}

Actions already taken:
{state.actions}

Recent observations:
{state.observations[-3:]}

AVAILABLE ACTIONS

GET_ORDER
GET_DELIVERY
EXTRACT_GRIEVANCES
GET_EVIDENCE
RUN_JUDGES
BUILD_CONSENSUS
APPLY_POLICY
SETTLE
ESCALATE
STOP

RULES

- Choose exactly ONE action.
- Use the current state to determine what should happen next.
- Do not calculate refunds yourself.
- Do not invent evidence.
- Do not modify business policy.
- If an action failed or an order was not found, use the observation
  to determine whether the case can continue.
- Do not proceed to dependent actions after a failed prerequisite.
- ESCALATE only when the case cannot safely continue.
- STOP only when processing is complete.

Return JSON only:

{{
    "action": "GET_ORDER"
}}
"""

        try:
            response = self.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a customer-support orchestration agent. "
                            "Choose the next action using only the available "
                            "state and allowed actions."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                case_id=state.case_id,
                order_id=state.order_id,
            )
        except (NotFoundError, PermissionDeniedError) as error:
            raise ValueError(
                f"Configured OPEN_AI_MODEL '{self.model}' is unavailable "
                "to this OpenAI account."
            ) from error

        content = response.choices[0].message.content

        if content is None:
            raise ValueError(
                "LLM returned invalid AgentDecision: response content was empty."
            )

        try:
            return AgentDecision.model_validate_json(content)
        except ValueError as error:
            raise ValueError(
                "LLM returned invalid AgentDecision: response did not match "
                "the expected structured output."
            ) from error