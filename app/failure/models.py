from enum import Enum


class FailureType(str, Enum):
    TRANSIENT = "TRANSIENT"
    PERMANENT = "PERMANENT"
    UNKNOWN = "UNKNOWN"


class FailureAction(str, Enum):
    RETRY = "RETRY"
    CONTINUE = "CONTINUE"
    ESCALATE = "ESCALATE"
