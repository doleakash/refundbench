import requests

from config.settings import get_settings


class RefundBenchAPIClient:
	def __init__(self, base_url: str | None = None):
		self.base_url = (
			base_url or get_settings().backend_url
		).rstrip("/")

	def get_orders(self) -> list[str]:
		response = requests.get(
			f"{self.base_url}/orders",
			timeout=3,
		)
		response.raise_for_status()
		return response.json()

	def create_case(
		self,
		order_id: str,
		customer_message: str,
	) -> dict:
		response = requests.post(
			f"{self.base_url}/cases",
			json={
				"order_id": order_id,
				"customer_message": customer_message,
			},
			timeout=300,
		)
		response.raise_for_status()
		return response.json()

	def get_case(self, case_id: str) -> dict:
		response = requests.get(
			f"{self.base_url}/cases/{case_id}",
			timeout=30,
		)
		response.raise_for_status()
		return response.json()

	def accept_refund(self, case_id: str) -> dict:
		response = requests.post(
			f"{self.base_url}/cases/{case_id}/refund/accept",
			timeout=30,
		)
		response.raise_for_status()
		return response.json()
