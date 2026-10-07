from fastapi import HTTPException

from app.application.cases.errors import (
	CaseApplicationError,
	CaseNotFound,
	InvalidCaseRequest,
	RefundNotAcceptable,
)


def to_http_exception(error: CaseApplicationError) -> HTTPException:
	if isinstance(error, CaseNotFound):
		return HTTPException(status_code=404, detail="Case not found.")
	if isinstance(error, InvalidCaseRequest):
		return HTTPException(
			status_code=422,
			detail="Order ID and customer complaint are required.",
		)
	if isinstance(error, RefundNotAcceptable):
		return HTTPException(
			status_code=409,
			detail="This case has no approved refund proposal to accept.",
		)
	raise TypeError(f"Unhandled application error: {type(error).__name__}")
