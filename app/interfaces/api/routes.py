import logging

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.application.cases.errors import (
	CaseNotFound,
	InvalidCaseRequest,
	RefundNotAcceptable,
)
from app.application.cases.service import CaseApplicationService
from app.infrastructure.order_repository import OrderDataError
from app.interfaces.api.errors import to_http_exception
from app.interfaces.api.mappers import case_to_response
from app.interfaces.api.schemas import (
	CaseRequest,
	CaseResult,
	RefundAcceptanceResult,
)

logger = logging.getLogger(__name__)


def create_api_app(service: CaseApplicationService) -> FastAPI:
	app = FastAPI(title="RefundBench API")
	app.add_middleware(
		CORSMiddleware,
		allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
		allow_credentials=False,
		allow_methods=["GET", "POST"],
		allow_headers=["*"],
	)

	router = APIRouter()

	@router.get("/orders", response_model=list[str])
	def list_orders() -> list[str]:
		try:
			return service.list_order_ids()
		except OrderDataError:
			logger.exception("Unable to load available order IDs")
			raise HTTPException(
				status_code=500,
				detail="Unable to load available orders.",
			) from None

	@router.post("/cases", response_model=CaseResult)
	def create_case(request: CaseRequest) -> CaseResult:
		try:
			state = service.process_case(
				order_id=request.order_id,
				customer_message=request.customer_message,
			)
		except InvalidCaseRequest as error:
			raise to_http_exception(error) from None
		except Exception:
			logger.exception("Agent failed while processing a case")
			raise HTTPException(
				status_code=500,
				detail="Agent could not safely process this case.",
			) from None
		return case_to_response(state)

	@router.get("/cases/{case_id}", response_model=CaseResult)
	def get_case(case_id: str) -> CaseResult:
		try:
			state = service.get_case(case_id)
		except CaseNotFound as error:
			raise to_http_exception(error) from None
		return case_to_response(state)

	@router.post(
		"/cases/{case_id}/refund/accept",
		response_model=RefundAcceptanceResult,
	)
	def accept_refund(case_id: str) -> RefundAcceptanceResult:
		try:
			refund = service.accept_refund(case_id)
		except (CaseNotFound, RefundNotAcceptable) as error:
			raise to_http_exception(error) from None
		except Exception:
			logger.exception(
				"Refund processing failed for case %s",
				case_id,
			)
			raise HTTPException(
				status_code=502,
				detail="Refund processing failed. You may retry the request.",
			) from None
		return RefundAcceptanceResult(
			case_id=case_id,
			order_id=refund.order_id,
			status=refund.status,
			amount=refund.amount,
			idempotency_key=refund.idempotency_key,
		)

	app.include_router(router)
	return app
