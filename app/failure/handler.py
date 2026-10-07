from app.failure.models import FailureAction, FailureType


def handle_failure(failure_type: FailureType) -> FailureAction:
    if failure_type is FailureType.TRANSIENT:
        return FailureAction.RETRY

    if failure_type is FailureType.PERMANENT:
        return FailureAction.CONTINUE

    return FailureAction.ESCALATE