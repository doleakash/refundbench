from app.domain.models import Order, Delivery
from app.infrastructure.order_repository import JSONOrderRepository
from config.settings import get_settings

_order_repository = JSONOrderRepository(get_settings().data_dir)
FAIL_GET_ORDER = False


def get_order(order_id: str) -> Order | None:
	if FAIL_GET_ORDER:
		raise TimeoutError("Order service timed out")
	return _order_repository.get_order(order_id)


def get_delivery(order_id: str) -> Delivery | None:
	return _order_repository.get_delivery(order_id)
