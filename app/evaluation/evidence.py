from __future__ import annotations

from datetime import datetime

from app.domain.models import (
	Evidence,
	Grievance,
	GrievanceType,
	Order,
	Delivery,
)


def get_evidence(
	grievance: Grievance,
	order: Order,
	delivery: Delivery | None,
) -> Evidence:
	if grievance.type == GrievanceType.LATE_DELIVERY:

		if delivery is None:
			return Evidence(
				grievance_id=grievance.grievance_id,
				source="DELIVERY_SYSTEM",
				facts=["Delivery data unavailable"],
			)

		expected = datetime.fromisoformat(
			order.expected_delivery_at
		)

		actual = datetime.fromisoformat(
			delivery.delivered_at
		)

		delay_minutes = (
							actual - expected
						).total_seconds() / 60

		return Evidence(
			grievance_id=grievance.grievance_id,
			source="DELIVERY_SYSTEM",
			facts=[
				f"Expected delivery: {order.expected_delivery_at}",
				f"Actual delivery: {delivery.delivered_at}",
				f"Delay: {delay_minutes:.0f} minutes",
			],
		)

	if grievance.type == GrievanceType.MISSING_ITEMS:

		if delivery is None:
			return Evidence(
				grievance_id=grievance.grievance_id,
				source="ORDER_DELIVERY_DATA",
				facts=["Delivery data unavailable"],
			)

		delivered_by_item = {
			item.item_id: item.quantity
			for item in delivery.items
		}

		facts = []

		for item in order.items:
			delivered_quantity = delivered_by_item.get(
				item.item_id,
				0,
			)

			missing_quantity = max(
				item.quantity - delivered_quantity,
				0,
			)

			facts.append(
				f"Item: {item.name}, "
				f"ordered: {item.quantity}, "
				f"delivered: {delivered_quantity}, "
				f"missing: {missing_quantity}, "
				f"unit price: ₹{item.unit_price:.2f}"
			)

		return Evidence(
			grievance_id=grievance.grievance_id,
			source="ORDER_DELIVERY_DATA",
			facts=facts,
		)

	if grievance.type == GrievanceType.LEAKED_ITEM:
		return Evidence(
			grievance_id=grievance.grievance_id,
			source="CUSTOMER_IMAGE",
			facts=[
				"Customer image required to verify spillage"
			],
			requires_customer_input=True,
		)

	return Evidence(
		grievance_id=grievance.grievance_id,
		source="UNKNOWN",
		facts=["No evidence strategy available"],
	)
