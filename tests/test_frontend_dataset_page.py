import json
from pathlib import Path

from app.domain.models import Delivery, Order
from app.evaluation.benchmark import GoldenCase

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "frontend" / "refundbench.html"
DATA = ROOT / "data"


def _read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def test_html_page_loads_fixture_sources_separately_from_benchmark_results():
    html = HTML.read_text(encoding="utf-8")
    assert 'dashboard: &quot;../data/benchmark_results/dashboard.json&quot;' in html
    assert 'cases: &quot;../data/benchmark_results/case_results.json&quot;' in html
    assert "path:'../data/golden_set.json'" in html
    assert "path:'../data/orders.json'" in html
    assert "path:'../data/deliveries.json'" in html
    assert 'id="view-dataset"' in html
    assert "fixtureErrors={};fixtureData={};" in html


def test_fixture_counts_models_and_primary_category_counts_reconcile():
    cases = _read("golden_set.json")
    orders = _read("orders.json")
    deliveries = _read("deliveries.json")
    dashboard = _read("benchmark_results/dashboard.json")
    saved_results = _read("benchmark_results/case_results.json")

    assert all(GoldenCase.model_validate(case) for case in cases)
    assert all(Order.model_validate(order) for order in orders)
    assert all(Delivery.model_validate(delivery) for delivery in deliveries)
    assert len(saved_results) == dashboard["case_count"]
    assert len(cases) == len({case["case_id"] for case in cases})

    category_counts = {}
    for case in cases:
        category = case.get("category") or "Unclassified"
        category_counts[category] = category_counts.get(category, 0) + 1
    assert sum(category_counts.values()) == len(cases)
    assert len({order["order_id"] for order in orders}) == len(orders)
    assert all(delivery["order_id"] in {order["order_id"] for order in orders} for delivery in deliveries)


def test_dataset_controls_have_handlers_for_selection_and_export_without_run_validation():
    html = HTML.read_text(encoding="utf-8")
    for control_id in (
        "dataset-catalog-search",
        "download-schema",
        "coverage-category-filter",
        "coverage-case-select",
    ):
        assert f'id="{control_id}"' in html
    assert 'id="run-dataset-validation"' not in html
    assert "Run Data Validation" not in html
    assert 'aria-label="Dataset overview"' in html
    assert html.count('class="dataset-card"><div class="dataset-label">') >= 3
    assert "Data Validation" not in html
    assert "createObjectURL(new Blob([csv]" in html
    assert "renderSchemaExplorer()" in html
