import pytest
from types import SimpleNamespace

from openai import APIConnectionError
from openai._base_client import httpx2

from app.agent.state import Action, AgentState
from app.infrastructure.llm_model import LLMModel


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
    assert len(retry_events) == 1
    assert retry_events[0].case_id == "CASE-RETRY"
    assert retry_events[0].order_id == "ORD-123"


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
        (None, None, None),
        ("GROQ", None, "https://api.groq.com/openai/v1"),
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
    if provider is None:
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
    else:
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
    if expected_base_url is None:
        assert "base_url" not in captured
    else:
        assert captured["base_url"] == expected_base_url


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