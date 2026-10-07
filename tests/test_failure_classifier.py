from app.failure.classifier import classify_failure
from app.failure.models import FailureType


def test_timeout_is_transient():
    error = RuntimeError("Order service timeout")

    assert classify_failure(error) == FailureType.TRANSIENT


def test_service_unavailable_is_transient():
    error = RuntimeError("Order service unavailable")

    assert classify_failure(error) == FailureType.TRANSIENT


def test_invalid_request_is_permanent():
    error = ValueError("Invalid order ID")

    assert classify_failure(error) == FailureType.PERMANENT


def test_unknown_error():
    error = RuntimeError("Something unexpected happened")

    assert classify_failure(error) == FailureType.UNKNOWN