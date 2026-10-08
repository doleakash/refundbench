from enum import Enum

from pydantic import BaseModel, Field


class OrderItem(BaseModel):
	item_id: str
	name: str
	quantity: int
	unit_price: float


class Order(BaseModel):
	order_id: str
	customer_id: str
	expected_delivery_at: str
	status: str
	total_amount: float
	order_weight: float
	items: list[OrderItem]


class DeliveredItem(BaseModel):
	item_id: str
	quantity: int


class Delivery(BaseModel):
	order_id: str
	delivered_at: str
	status: str
	delivered_weight: float
	items: list[DeliveredItem] = Field(default_factory=list)


class GrievanceType(str, Enum):
	LATE_DELIVERY = "LATE_DELIVERY"
	MISSING_ITEMS = "MISSING_ITEMS"
	LEAKED_ITEM = "LEAKED_ITEM"
	FOOD_QUALITY = "FOOD_QUALITY"
	WRONG_ITEM = "WRONG_ITEM"


class CustomerIntent(str, Enum):
	REFUND_REQUEST = "REFUND_REQUEST"
	REFUND_STATUS = "REFUND_STATUS"
	ORDER_SUPPORT = "ORDER_SUPPORT"
	DELIVERY_SUPPORT = "DELIVERY_SUPPORT"
	GENERAL_SUPPORT = "GENERAL_SUPPORT"


class Grievance(BaseModel):
	grievance_id: str
	type: GrievanceType
	claim: str
	raw_claims: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
	grievance_id: str
	source: str
	facts: list[str]
	requires_customer_input: bool = False
