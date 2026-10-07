from app.failure.models import FailureType


def classify_failure(error: Exception) -> FailureType:
    message = str(error).lower()

    if any(
        keyword in message
        for keyword in [
            "timeout",
            "timed out",
            "service unavailable",
            "connection reset",
        ]
    ):
        return FailureType.TRANSIENT

    if any(
        keyword in message
        for keyword in [
            "invalid",
            "bad request",
        ]
    ):
        return FailureType.PERMANENT

    return FailureType.UNKNOWN