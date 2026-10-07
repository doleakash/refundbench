class CaseApplicationError(Exception):
	"""Base class for expected case-use-case failures."""


class CaseNotFound(CaseApplicationError):
	pass


class InvalidCaseRequest(CaseApplicationError):
	pass


class RefundNotAcceptable(CaseApplicationError):
	pass
