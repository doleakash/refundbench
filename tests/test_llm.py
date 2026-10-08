import pytest
from types import SimpleNamespace

from openai import APIConnectionError
from openai._base_client import httpx2

from app.agent.state import Action, AgentState
from app.infrastructure.llm_model import LLMModel


@pytest.mark.parametrize(
    ("usage", "expected_tokens"),
    [
        (
            SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
            ),
            (11, 7, 18),
        ),
        (None, (None, None, None)),
    ],
)
def test_llm_complete_records_provider_token_usage(
    monkeypatch,
    usage,
    expected_tokens,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPEN_AI_MODEL", "test-model")
    response = SimpleNamespace(usage=usage)
    model = LLMModel()
    model.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response),
        ),
    )
    timing_events = []
    monkeypatch.setattr(
        "app.infrastructure.llm_model.log_timing_event",
        timing_events.append,
    )

    result, event = model.complete(
        [],
        case_id="CASE-USAGE",
        order_id="ORD-123",
    )

    assert len(timing_events) == 1
    assert result is response
    assert event.stage == "LLM_CALL"
    assert event.success is True
    assert event.model == "test-model"
    assert event.case_id == "CASE-USAGE"
    assert event.order_id == "ORD-123"
    assert (
        event.prompt_tokens,
        event.completion_tokens,
        event.total_tokens,
    ) == expected_tokens


def test_llm_retries_transient_failure_once(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPEN_AI_MODEL", "test-model")

    calls = 0
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"action":"GET_ORDER"}')
            )
        ]
    )

    def create(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise APIConnectionError(
                request=httpx2.Request(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                )
            )
        return response

    model = LLMModel()
    model.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
    )
    retry_events = []
    monkeypatch.setattr(
        "app.infrastructure.llm_model.log_timing_event",
        retry_events.append,
    )
    monkeypatch.setattr(
        "app.infrastructure.llm_model.sleep",
        lambda _: None,
    )

    result = model.decide(
        AgentState(
            customer_message="My order arrived late.",
            order_id="ORD-123",
            case_id="CASE-RETRY",
        )
    )

    assert result.action is Action.GET_ORDER
    assert calls == 2
    assert len(retry_events) == 2
    assert retry_events[0].case_id == "CASE-RETRY"
    assert retry_events[0].order_id == "ORD-123"
    assert retry_events[0].stage == "LLM_RETRY:1"
    assert retry_events[1].stage == "LLM_CALL"
    assert retry_events[1].prompt_tokens is None


def test_llm_invalid_decision_output_is_not_retried(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPEN_AI_MODEL", "test-model")

    calls = 0
    model = LLMModel()

    def create(**kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"action":"INVALID"}')
                )
            ]
        )

    model.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
    )

    with pytest.raises(ValueError, match="LLM returned invalid AgentDecision") as error:
        model.decide(
            AgentState(
                customer_message="My order arrived late.",
                order_id="ORD-123",
                case_id="CASE-INVALID-OUTPUT",
            )
        )

    assert calls == 1
    assert error.value.__cause__ is not None


@pytest.mark.parametrize(
    ("provider", "configured_base_url", "expected_base_url"),
    [
        ("OPENAI", None, "https://api.openai.com/v1"),
        ("XAI", "https://api.x.ai/v1", "https://api.x.ai/v1"),
        ("GROQ", "https://groq.example/v1", "https://groq.example/v1"),
    ],
)
def test_llm_client_uses_configured_provider_endpoint(
    monkeypatch,
    provider,
    configured_base_url,
    expected_base_url,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPEN_AI_MODEL", "test-model")
    monkeypatch.setenv("LLM_PROVIDER", provider)
    if configured_base_url is None:
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("LLM_BASE_URL", configured_base_url)

    captured = {}

    def openai_client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("app.infrastructure.llm_model.OpenAI", openai_client)

    LLMModel()

    assert captured["api_key"] == "test-key"
    assert captured["timeout"] > 0
    assert captured["max_retries"] == 0
    assert captured["base_url"] == expected_base_url


def test_llm_provider_is_informational(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ARBITRARY_PROVIDER")

    from config.settings import get_settings

    assert get_settings().llm_provider == "ARBITRARY_PROVIDER"


def test_open_ai_model_remains_required(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPEN_AI_MODEL", raising=False)

    with pytest.raises(ValueError, match="OPEN_AI_MODEL is not configured"):
        LLMModel()


def main():
    state = AgentState(
        customer_message="My order ORD-123 was delivered late. Can I get a refund?",
        order_id="ORD-123",
        case_id="CASE-001",
    )

    model = LLMModel()

    decision = model.decide(state)
    print(decision)


if __name__ == "__main__":
    main()