import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.domain.models import Delivery, Order
from app.evaluation.benchmark import load_golden_cases

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NEW_CATEGORY_COUNTS = {
    "delivery_fulfillment_edge": 20,
    "item_quantity_discrepancy": 15,
    "evidence_conflict_uncertainty": 15,
    "grievance_multi_issue": 10,
    "settlement_idempotency": 10,
    "consensus_escalation": 10,
    "boundary_adversarial": 10,
    "operational_failure_recovery": 10,
}
ORIGINAL_27_SHA256 = "8389323f65c40122a8a655b44c5ba5aa23d145072b15241e54fe6a4a430e199b"


def _json_data(filename):
    return json.loads((DATA / filename).read_text())


def test_fixture_schemas_ids_and_order_delivery_references():
    orders_data = _json_data("orders.json")
    deliveries_data = _json_data("deliveries.json")
    orders = [Order.model_validate(row) for row in orders_data]
    deliveries = [Delivery.model_validate(row) for row in deliveries_data]
    order_ids = [order.order_id for order in orders]
    delivery_ids = [delivery.order_id for delivery in deliveries]

    assert len(order_ids) == len(set(order_ids))
    assert set(delivery_ids) <= set(order_ids)
    duplicate_delivery_orders = {
        order_id for order_id, count in Counter(delivery_ids).items() if count > 1
    }
    assert duplicate_delivery_orders == {"ORD-144", "ORD-148"}

    order_by_id = {order.order_id: order for order in orders}
    for order in orders:
        if order.items:
            assert order.total_amount == sum(
                item.quantity * item.unit_price for item in order.items
            )
            assert all(item.quantity > 0 and item.unit_price >= 0 for item in order.items)
        assert datetime.fromisoformat(order.expected_delivery_at)

    for delivery in deliveries:
        assert delivery.status in {
            "DELIVERED",
            "PARTIALLY_DELIVERED",
            "IN_TRANSIT",
            "CANCELLED",
        }
        if delivery.delivered_at is not None:
            assert datetime.fromisoformat(delivery.delivered_at)
        if delivery.items:
            ordered_ids = {item.item_id for item in order_by_id[delivery.order_id].items}
            delivered_ids = {item.item_id for item in delivery.items}
            if delivery.order_id in {"ORD-138", "ORD-157"}:
                assert delivered_ids - ordered_ids
            else:
                assert delivered_ids <= ordered_ids
        assert all(item.quantity >= 0 for item in delivery.items)

    assert Counter(delivery_ids)["ORD-144"] == 2
    assert Counter(delivery_ids)["ORD-148"] == 2
    assert "ORD-147" not in delivery_ids
    assert len(orders) == 41
    assert len(deliveries) == 33

    # The repository intentionally returns the first record for duplicate-order fixtures.
    assert next(row.delivered_at for row in deliveries if row.order_id == "ORD-144") == (
        "2026-09-27T16:30:00"
    )


def test_golden_cases_are_sequential_preserve_originals_and_reference_data():
    raw_cases = _json_data("golden_set.json")
    cases = load_golden_cases()
    case_ids = [case.case_id for case in cases]
    assert len(cases) == 127
    assert case_ids == [f"CASE-GOLDEN-{number:03}" for number in range(1, 128)]

    original = json.dumps(raw_cases[:27], sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(original.encode()).hexdigest() == ORIGINAL_27_SHA256

    order_ids = {row["order_id"] for row in _json_data("orders.json")}
    for case in cases:
        # Two cases deliberately test malformed and whitespace-padded IDs.
        if case.case_id == "CASE-GOLDEN-108":
            assert case.order_id == "ord-140"
            assert case.expected["outcome"] == "ESCALATED"
        elif case.case_id == "CASE-GOLDEN-109":
            assert case.order_id == " ORD-140 "
        elif case.order_id == "ORD-999":
            assert case.case_id in {"CASE-GOLDEN-012", "CASE-GOLDEN-118"}
        else:
            assert case.order_id.strip() in order_ids
        assert case.expected["intent"]
        assert "grievances" in case.expected and "judgements" in case.expected
        assert "settlement" in case.expected and "escalation" in case.expected
        if case.case_id >= "CASE-GOLDEN-028":
            assert case.expected["outcome"] in {"RESOLVED", "ESCALATED", "DEFLECTED"}


def test_new_case_category_distribution_and_scenario_coverage():
    cases = load_golden_cases()
    new_cases = cases[27:]
    assert Counter(case.category for case in new_cases) == NEW_CATEGORY_COUNTS

    by_id = {case.case_id: case for case in cases}
    assert by_id["CASE-GOLDEN-016"].expected["settlement"] is None
    assert by_id["CASE-GOLDEN-024"].expected["outcome"] == "DEFLECTED"
    assert by_id["CASE-GOLDEN-017"].order_id == "ORD-135"
    assert by_id["CASE-GOLDEN-018"].order_id == "ORD-136"
    assert by_id["CASE-GOLDEN-019"].order_id == "ORD-139"
    assert by_id["CASE-GOLDEN-020"].order_id == "ORD-138"
    assert by_id["CASE-GOLDEN-025"].order_id == "ORD-139"
    assert by_id["CASE-GOLDEN-026"].order_id == "ORD-139"

    assert by_id["CASE-GOLDEN-028"].category == "delivery_fulfillment_edge"
    assert by_id["CASE-GOLDEN-047"].order_id == "ORD-144"
    assert by_id["CASE-GOLDEN-058"].order_id == "ORD-157"
    assert by_id["CASE-GOLDEN-088"].expected["settlement"]["refund_status"] == "PROCESSING"
    assert by_id["CASE-GOLDEN-097"].expected["settlement"]["decision"] == "ESCALATE"
    assert by_id["CASE-GOLDEN-108"].category == "boundary_adversarial"
    assert by_id["CASE-GOLDEN-118"].expected["outcome"] == "ESCALATED"

    # Existing refund state is seeded through the benchmark's supported precondition contract.
    for case_id, status, amount, order_id in [
        ("CASE-GOLDEN-088", "PROCESSING", 100, "ORD-140"),
        ("CASE-GOLDEN-089", "REFUNDED", 100, "ORD-140"),
        ("CASE-GOLDEN-090", "PROPOSED", 100, "ORD-140"),
        ("CASE-GOLDEN-091", "PROCESSING", 350, "ORD-139"),
        ("CASE-GOLDEN-093", "PROCESSING", 50, "ORD-140"),
        ("CASE-GOLDEN-094", "REFUNDED", 350, "ORD-139"),
        ("CASE-GOLDEN-095", "PROPOSED", 350, "ORD-139"),
        ("CASE-GOLDEN-125", "PROCESSING", 100, "ORD-140"),
    ]:
        precondition = by_id[case_id].preconditions.existing_refund
        assert (precondition.status, precondition.amount, precondition.idempotency_key) == (
            status,
            amount,
            f"refund:{order_id}",
        )


def test_claim_scoped_curry_fixture_has_only_the_claimed_discrepancy():
    orders = {row["order_id"]: row for row in _json_data("orders.json")}
    delivery = next(row for row in _json_data("deliveries.json") if row["order_id"] == "ORD-139")
    order = Order.model_validate(orders["ORD-139"])
    delivery = Delivery.model_validate(delivery)
    delivered = {item.item_id: item.quantity for item in delivery.items}
    missing = {
        item.name: item.quantity - delivered.get(item.item_id, 0)
        for item in order.items
    }
    assert missing == {"Chicken Curry": 1, "Naan": 0}


def test_expected_item_refunds_match_recorded_discrepancy_and_item_prices():
    orders = {row["order_id"]: row for row in _json_data("orders.json")}
    deliveries = _json_data("deliveries.json")
    # These cases deliberately exercise exact item-level discrepancies supported by the fixtures.
    expected_claim_scoped = {
        "CASE-GOLDEN-019": 350,
        "CASE-GOLDEN-026": 350,
        "CASE-GOLDEN-048": 350,
        "CASE-GOLDEN-049": 410,
        "CASE-GOLDEN-050": 250,
        "CASE-GOLDEN-051": 2000,
        "CASE-GOLDEN-052": 2500,
    }
    cases = {case.case_id: case for case in load_golden_cases()}
    for case_id, amount in expected_claim_scoped.items():
        case = cases[case_id]
        order = Order.model_validate(orders[case.order_id])
        delivery = next(row for row in deliveries if row["order_id"] == case.order_id)
        delivered = {row["item_id"]: row["quantity"] for row in delivery["items"]}
        discrepancy_value = sum(
            max(0, item.quantity - delivered.get(item.item_id, 0)) * item.unit_price
            for item in order.items
        )
        assert discrepancy_value == amount
        assert case.expected["settlement"]["refund_amount"] == amount
