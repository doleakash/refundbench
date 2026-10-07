from app.failure.handler import handle_failure
from app.failure.models import FailureAction, FailureType


def test_transient_failure_should_retry():
    assert handle_failure(FailureType.TRANSIENT) == FailureAction.RETRY


def test_permanent_failure_should_continue():
    assert handle_failure(FailureType.PERMANENT) == FailureAction.CONTINUE


def test_unknown_failure_should_escalate():
    assert handle_failure(FailureType.UNKNOWN) == FailureAction.ESCALATE