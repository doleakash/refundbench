from dataclasses import dataclass

from app.agent.orchestrator import run_agent
from app.application.cases.repository import InMemoryCaseRepository
from app.application.cases.service import CaseApplicationService
from app.infrastructure.order_repository import JSONOrderRepository
from app.settlement.refund_ledger import RefundLedger
from app.settlement.refund_provider import MockRefundProvider
from app.settlement.refund_service import RefundService
from app.settlement.refund_workflow import RefundWorkflow
from app.settlement.settlement import Settlement
from app.interfaces.api.routes import create_api_app
from config.logging import configure_logging
from config.settings import Settings, get_settings


@dataclass
class ApplicationContainer:
	case_repository: InMemoryCaseRepository
	order_repository: JSONOrderRepository
	case_service: CaseApplicationService
	refund_workflow: RefundWorkflow


def build_container(settings: Settings | None = None) -> ApplicationContainer:
	settings = settings or get_settings()
	configure_logging(settings)

	case_repository = InMemoryCaseRepository()
	order_repository = JSONOrderRepository(settings.data_dir)
	ledger = RefundLedger()
	settlement = Settlement(ledger=ledger)
	refund_provider = MockRefundProvider()
	refund_service = RefundService(
		ledger=ledger,
		provider=refund_provider,
	)
	refund_workflow = RefundWorkflow(
		settlement=settlement,
		refund_service=refund_service,
	)
	case_service = CaseApplicationService(
		cases=case_repository,
		orders=order_repository,
		refund_workflow=refund_workflow,
		agent_runner=run_agent,
	)

	return ApplicationContainer(
		case_repository=case_repository,
		order_repository=order_repository,
		case_service=case_service,
		refund_workflow=refund_workflow,
	)


container = build_container()


def create_app():
	return create_api_app(container.case_service)
