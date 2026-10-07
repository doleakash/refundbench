import json
from pathlib import Path

from app.domain.models import Delivery, Order


class OrderDataError(Exception):
	"""Order or delivery data could not be read or validated."""


class JSONOrderRepository:
	def __init__(self, data_dir: Path):
		self.data_dir = data_dir

	def list_order_ids(self) -> list[str]:
		try:
			with (self.data_dir / "orders.json").open(encoding="utf-8") as file:
				return [
					order["order_id"]
					for order in json.load(file)
				]
		except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
			raise OrderDataError("Unable to load available order IDs.") from error

	def get_order(self, order_id: str) -> Order | None:
		for order in self._read_records("orders.json"):
			if order["order_id"] == order_id:
				return Order(**order)
		return None

	def get_delivery(self, order_id: str) -> Delivery | None:
		for delivery in self._read_records("deliveries.json"):
			if delivery["order_id"] == order_id:
				return Delivery(**delivery)
		return None

	def _read_records(self, filename: str) -> list[dict]:
		try:
			with (self.data_dir / filename).open(encoding="utf-8") as file:
				return json.load(file)
		except (OSError, json.JSONDecodeError, TypeError) as error:
			raise OrderDataError(
				f"Unable to load {filename}."
			) from error
