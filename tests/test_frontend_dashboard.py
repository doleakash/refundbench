import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from benchmark.dashboard_artifacts import (
	build_dashboard_artifact,
	load_saved_evaluation_results,
	write_dashboard_artifacts,
)
from benchmark.generate_dashboard import _default_prices_for_saved_run
from benchmark.models import BenchmarkResult, CasePerformance, FieldComparison
from frontend.app import (
	dashboard_artifact_empty_state,
	dashboard_cases_csv,
	load_dashboard_artifacts,
	truncate_display_text,
)


def test_dashboard_total_automation_rate_counts_resolved_and_deflected_outcomes():
	root = Path(__file__).resolve().parents[1]
	cases = json.loads(
		(root / "data" / "benchmark_results" / "case_results.json")
		.read_text(encoding="utf-8")
	)
	outcomes = Counter(case.get("actual", {}).get("outcome") for case in cases)
	classified = sum(outcomes[value] for value in ("RESOLVED", "DEFLECTED", "ESCALATED"))
	automated = outcomes["RESOLVED"] + outcomes["DEFLECTED"]

	assert (automated / classified * 100) == pytest.approx(79.53, abs=0.01)

	html = (root / "frontend" / "refundbench.html").read_text(encoding="utf-8")
	assert "const automatedOutcomes = terminalOutcomes.filter" in html
	assert "RESOLVED&quot;,&quot;DEFLECTED&quot;" in html
	assert "recorded outcomes resolved or deflected" in html


def make_result(
	case_id,
	expected_decision,
	expected_amount,
	expected_escalation,
	actual_decision,
	actual_amount,
	actual_escalation,
	*,
	latency=1000,
	prompt_tokens=10,
	completion_tokens=5,
):
	expected = {
		"intent": "REFUND_REQUEST",
		"outcome": None,
		"grievance_types": [],
		"judgements": [],
		"settlement.decision": expected_decision,
		"settlement.refund_amount": expected_amount,
		"escalation": expected_escalation,
	}
	actual = {
		"intent": "REFUND_REQUEST",
		"outcome": (
			"ESCALATED"
			if actual_escalation or actual_decision == "ESCALATE"
			else "RESOLVED"
		),
		"grievance_types": [],
		"judgements": [],
		"settlement.decision": actual_decision,
		"settlement.refund_amount": actual_amount,
		"escalation": actual_escalation,
	}
	return BenchmarkResult(
		case_id=case_id,
		passed=(expected["outcome"] is None or expected["outcome"] == actual["outcome"])
		and all(expected[field] == actual[field] for field in (
			"settlement.decision", "settlement.refund_amount", "escalation"
		)),
		comparisons=[
			FieldComparison(
				field=field,
				expected=expected_value,
				actual=actual[field],
				passed=expected_value == actual[field],
			)
			for field, expected_value in expected.items()
		],
		performance=CasePerformance(
			latency_ms=latency,
			prompt_token_count=prompt_tokens,
			completion_token_count=completion_tokens,
		),
		category="late_delivery",
		customer_message=f"Customer issue for {case_id}",
		actual_response="We reviewed your order and explained the resolution.",
	)


def test_dashboard_metrics_classify_escalations_as_not_automatically_approved():
	results = [
		make_result(
			"CASE-TP", "AUTO_APPROVED", 640, False,
			"AUTO_APPROVED", 640, False,
		),
		make_result(
			"CASE-FP", "AUTO_APPROVED", 0, False,
			"AUTO_APPROVED", 320, False,
		),
		make_result(
			"CASE-REVIEW", "AUTO_APPROVED", 640, False,
			"ESCALATE", 640, True,
		),
	]

	dashboard, cases = build_dashboard_artifact(
		results,
		input_cost_per_million_tokens=0.20,
		output_cost_per_million_tokens=1.20,
	)

	metrics = dashboard["metrics"]
	assert metrics["automation_rate"] == 2 / 3
	assert metrics["escalation_count"] == 1
	assert metrics["decision_confusion_matrix"] == {
		"true_positive": 1,
		"true_negative": 0,
		"false_positive": 1,
		"false_negative": 1,
	}
	assert metrics["decision_confusion_matrix_denominator"] == 3
	assert metrics["agent_escalations_excluded_from_confusion_matrix"] == 0
	assert metrics["decision_confusion_matrix_excluded_case_count"] == 0
	assert metrics["estimated_llm_cost"] == 3 * (10 * 0.20 + 5 * 1.20) / 1_000_000
	assert cases[1]["expected"]["settlement.refund_amount"] == 0
	assert cases[1]["actual"]["settlement.refund_amount"] == 320
	assert cases[0]["simulated_csat"] == 5


def test_processing_outcomes_are_classified_as_not_automatically_approved():
	result = make_result(
		"CASE-PROCESSING",
		"PROCESSING",
		100,
		False,
		"PROCESSING",
		100,
		False,
	)

	dashboard, cases = build_dashboard_artifact([result])

	assert cases[0]["expected_refund_approved"] is False
	assert cases[0]["actual_refund_approved"] is False
	assert dashboard["metrics"]["decision_confusion_matrix"] == {
		"true_positive": 0,
		"true_negative": 1,
		"false_positive": 0,
		"false_negative": 0,
	}
	assert dashboard["metrics"]["incorrect_refund_decisions"] == {
		"false_positives": 0,
		"false_negatives": 0,
		"false_positive_rate": 0.0,
		"false_negative_rate": None,
	}


def test_recent_case_display_truncates_long_text_without_mutating_input():
	message = "A long customer issue " * 8

	truncated = truncate_display_text(message, 32)

	assert len(truncated) <= 32
	assert truncated.endswith("…")
	assert message == "A long customer issue " * 8


def test_dashboard_csv_exports_case_data_with_csv_escaping():
	csv_text = dashboard_cases_csv([{
	    "case_id": "CASE-1",
	    "customer_issue": 'Package was "damaged", please refund',
	    "display_actual_decision": "AUTO_APPROVED",
	    "display_expected_outcome": "AUTO_APPROVED · 100 refund",
	    "display_latency": "1250 ms",
	    "display_simulated_csat": "5.0 / 5",
	}])

	rows = list(csv.DictReader(csv_text.splitlines()))

	assert rows == [{
	    "Case ID": "CASE-1",
	    "Customer Message": 'Package was "damaged", please refund',
	    "Agent Decision": "AUTO_APPROVED",
	    "Expected Outcome": "AUTO_APPROVED · 100 refund",
	    "Resolution Time": "1250 ms",
	    "Simulated CSAT": "5.0 / 5",
	}]


def test_dashboard_artifact_writer_and_frontend_load_saved_json(tmp_path):
	result = make_result(
		"CASE-1", "AUTO_APPROVED", 640, False,
		"AUTO_APPROVED", 640, False,
	)
	case_path, dashboard_path = write_dashboard_artifacts(
		[result],
		tmp_path,
		input_cost_per_million_tokens=0.20,
		output_cost_per_million_tokens=1.20,
		cost_currency="EUR",
	)

	dashboard, cases = load_dashboard_artifacts(dashboard_path, case_path)

	assert dashboard["assumptions"]["cost_currency"] == "EUR"
	assert {
		"passed",
		"execution_status",
		"execution_error",
		"expected",
		"actual",
		"latency_ms",
		"llm_call_count",
		"simulated_csat",
	}.issubset(cases[0])
	assert "workflow_outcome_unavailable_count" in dashboard["metrics"]
	assert "workflow_outcome_distribution" in dashboard["metrics"]
	assert dashboard["metrics"]["estimated_net_savings"] is not None
	assert cases[0]["customer_issue"] == "Customer issue for CASE-1"
	assert cases[0]["display_actual_decision"] == "AUTO_APPROVED"
	assert dashboard["historical_trend"] is None
	assert json.loads(dashboard_path.read_text())["case_count"] == 1


def test_dashboard_artifact_writer_creates_parent_directory(tmp_path):
	result = make_result(
		"CASE-1", "AUTO_APPROVED", 640, False,
		"AUTO_APPROVED", 640, False,
	)
	output_dir = tmp_path / "nested" / "artifacts"

	case_path, dashboard_path = write_dashboard_artifacts([result], output_dir)

	assert case_path.is_file()
	assert dashboard_path.is_file()
	assert json.loads(dashboard_path.read_text())["case_count"] == 1


def test_dashboard_artifact_is_deterministic_for_fixed_inputs():
	result = make_result(
		"CASE-1", "AUTO_APPROVED", 640, False,
		"AUTO_APPROVED", 640, False,
	)

	first = build_dashboard_artifact(
		[result],
		input_cost_per_million_tokens=0.20,
		output_cost_per_million_tokens=1.20,
	)
	second = build_dashboard_artifact(
		[result],
		input_cost_per_million_tokens=0.20,
		output_cost_per_million_tokens=1.20,
	)

	assert first == second


def test_dashboard_missing_artifact_state_explains_generation_command():
	message = dashboard_artifact_empty_state()

	assert "`python -m benchmark.generate_dashboard`" in message
	assert "run the benchmark separately first" in message
	assert "`streamlit run app.py`" in message
	assert "never runs it" in message


def test_saved_report_is_recalculated_without_running_evaluation(tmp_path):
	golden = [{
		"case_id": "CASE-GOLDEN-001",
		"category": "delivery",
		"order_id": "ORDER-1",
		"customer_message": "A late delivery",
		"expected": {
			"intent": "REFUND_REQUEST",
			"grievances": [{"type": "LATE_DELIVERY", "claim": "Late"}],
			"judgements": [],
			"settlement": {
				"decision": "AUTO_APPROVED",
				"refund_amount": 100,
			},
			"escalation": False,
			"answer": "Expected response.",
		},
	}]
	golden_path = tmp_path / "golden.json"
	golden_path.write_text(json.dumps(golden), encoding="utf-8")
	report_path = tmp_path / "output.txt"
	trace = {
		"case_id": "TRACE-1",
		"outcome": "RESOLVED",
		"response": "Your refund has been approved after review.",
		"timing_events": [
			{
				"stage": "TOTAL_AGENT",
				"duration_ms": 1250,
				"model": None,
			},
			{
				"stage": "LLM_CALL",
				"model": "gpt-5.6-luna",
				"prompt_tokens": 12,
				"completion_tokens": 5,
				"total_tokens": 17,
			},
		],
	}
	report_path.write_text(
		json.dumps(trace, indent=2)
		+ "\nCASE-GOLDEN-001  PASS\n"
		+ "  CASE-GOLDEN-001 end-to-end latency: 1250.00 ms\n",
		encoding="utf-8",
	)

	results = load_saved_evaluation_results(report_path, golden_path)

	assert len(results) == 1
	assert results[0].passed is True
	assert results[0].performance.prompt_token_count == 12
	assert results[0].performance.completion_token_count == 5
	assert results[0].performance.latency_ms == 1250
	assert results[0].actual_response == trace["response"]
	comparisons = {item.field: item for item in results[0].comparisons}
	assert comparisons["settlement.decision"].actual == "AUTO_APPROVED"
	assert comparisons["outcome"].actual == "RESOLVED"


def test_dashboard_artifact_counts_escalation_in_matrix_and_marks_unknown_cost():
	result = make_result(
		"CASE-UNKNOWN", "AUTO_APPROVED", 640, False,
		"ESCALATE", 640, True, prompt_tokens=None,
	)
	dashboard, _ = build_dashboard_artifact([result])

	assert dashboard["metrics"]["estimated_llm_cost"] is None
	assert dashboard["metrics"]["estimated_net_savings"] is None
	assert dashboard["metrics"]["decision_confusion_matrix"] == {
		"true_positive": 0,
		"true_negative": 0,
		"false_positive": 0,
		"false_negative": 1,
	}
	assert dashboard["metrics"][
		"ambiguous_escalated_cases_excluded_from_confusion_matrix"
	] == 0


def test_default_token_prices_require_matching_saved_model_provider_endpoint():
	matching_report = json.dumps({
		"case_id": "CASE-1",
		"timing_events": [{
			"stage": "LLM_CALL",
			"model": "gpt-5.6-luna",
			"provider": "OPENAI",
			"endpoint": "https://api.openai.com/v1",
			"prompt_tokens": 100,
		}],
	}, indent=2)
	mismatched_report = json.dumps({
		"case_id": "CASE-1",
		"timing_events": [{
			"stage": "LLM_CALL",
			"model": "gpt-5.6-luna",
			"provider": "GROQ",
			"endpoint": "https://api.groq.com/openai/v1",
			"prompt_tokens": 100,
		}],
	}, indent=2)

	assert _default_prices_for_saved_run(matching_report) == (0.20, 1.20)
	assert _default_prices_for_saved_run(mismatched_report) == (None, None)
