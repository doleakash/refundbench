import ast
import json
import re
from collections import Counter
from math import isfinite
from pathlib import Path
from typing import Any

from benchmark.models import (
	BenchmarkResult,
	CasePerformance,
	FieldComparison,
	GoldenCase,
)
from benchmark.metrics import (
	automatic_refund_label,
	calculate_business_impact_metrics,
	calculate_decision_metrics,
	calculate_metrics,
	calculate_performance_metrics,
)


def _actual_values(result: BenchmarkResult) -> dict[str, Any]:
	return {
		comparison.field: comparison.actual
		for comparison in result.comparisons
	}


def _expected_values(result: BenchmarkResult) -> dict[str, Any]:
	return {
		comparison.field: comparison.expected
		for comparison in result.comparisons
	}


def load_saved_evaluation_results(
	report_path: Path,
	golden_set_path: Path,
) -> list[BenchmarkResult]:
	"""Rebuild benchmark results from a saved runner report without evaluation."""
	report = report_path.read_text(encoding="utf-8")
	with golden_set_path.open(encoding="utf-8") as golden_file:
		golden_cases = [
			GoldenCase.model_validate(case)
			for case in json.load(golden_file)
		]

	trace_starts = list(
		re.finditer(r'(?m)^\{\s*\n\s+"case_id":', report)
	)
	decoder = json.JSONDecoder()
	traces = [
		decoder.raw_decode(report, match.start())[0]
		for match in trace_starts
	]
	statuses = list(
		re.finditer(
			r"(?m)^(CASE-GOLDEN-\d+)\s+(PASS|FAIL)\s*$",
			report,
		)
	)
	if not traces or len(traces) != len(statuses) or len(traces) != len(golden_cases):
		raise ValueError(
			"Saved evaluation must contain one trace and one PASS/FAIL "
			"record for each golden case."
		)
	if any(
		status.group(1) != golden.case_id
		for status, golden in zip(statuses, golden_cases)
	):
		raise ValueError(
			"Saved evaluation case records do not match the golden-set order."
		)
	report_latencies = {
		match.group(1): float(match.group(2))
		for match in re.finditer(
			r"(?m)^\s+(CASE-GOLDEN-\d+) end-to-end latency:\s+"
			r"([0-9]+(?:\.[0-9]+)?) ms$",
			report,
		)
	}

	results = []
	for index, (golden, trace, status) in enumerate(
		zip(golden_cases, traces, statuses)
	):
		expected = {
			"intent": golden.expected.intent,
			"outcome": golden.expected.outcome,
			"grievance_types": [
				item.type for item in golden.expected.grievances
			],
			"judgements": [
				item.model_dump()
				for item in golden.expected.judgements
			],
			"escalation": golden.expected.escalation,
		}
		if golden.expected.settlement is not None:
			expected["settlement.decision"] = golden.expected.settlement.decision
			expected["settlement.refund_amount"] = (
				golden.expected.settlement.refund_amount
			)
		if (
			golden.expected.settlement is not None
			and golden.expected.settlement.refund_status is not None
		):
			expected["settlement.refund_status"] = (
				golden.expected.settlement.refund_status
			)
		if (
			golden.expected.settlement is not None
			and golden.expected.settlement.idempotency_key is not None
		):
			expected["settlement.idempotency_key"] = (
				golden.expected.settlement.idempotency_key
			)

		block_end = (
			statuses[index + 1].start()
			if index + 1 < len(statuses)
			else len(report)
		)
		actual = expected.copy()
		actual["outcome"] = trace.get("outcome")
		mismatches = list(
			re.finditer(
				r"(?m)^\s+MISMATCH \| ([^|]+) \| expected=(.*?) \| actual=(.*)$",
				report[status.end():block_end],
			)
		)
		for mismatch in mismatches:
			field = mismatch.group(1).strip()
			if field not in expected:
				raise ValueError(
					f"Unsupported comparison field in saved report: {field}"
				)
			try:
				actual[field] = ast.literal_eval(mismatch.group(3))
			except (ValueError, SyntaxError) as error:
				raise ValueError(
					f"Could not parse actual value for {field} in saved report."
				) from error
		if status.group(2) == "FAIL" and not mismatches:
			raise ValueError(
				f"Saved report marks {golden.case_id} as failed without "
				"comparison details."
			)
		comparisons = [
			FieldComparison(
				field=field,
				expected=expected_value,
				actual=actual[field],
				passed=(
					expected_value is None or actual[field] == expected_value
					if field == "outcome"
					else actual[field] == expected_value
				),
			)
			for field, expected_value in expected.items()
		]
		outcome_passed = (
			golden.expected.outcome is None
			or actual["outcome"] == golden.expected.outcome
		)
		settlement_passed = all(
			item.passed
			for item in comparisons
			if item.field.startswith("settlement.")
		)
		passed = outcome_passed and settlement_passed
		timing_events = trace.get("timing_events")
		if not isinstance(timing_events, list):
			raise ValueError(
				f"Saved trace for {golden.case_id} has no timing event list."
			)
		llm_events = [
			event for event in timing_events
			if event.get("model") is not None
			and not event.get("stage", "").startswith("LLM_RETRY:")
			and (
				event.get("provider") is not None
				or event.get("stage") == "LLM_CALL"
				or (
					event.get("stage", "").startswith("LLM_JUDGE:")
					and event.get("success") is not False
				)
				or (
					event.get("stage") == "LLM_EXTRACT_GRIEVANCES"
					and event.get("success") is not False
				)
			)
		]
		instrumentation_available = bool(timing_events)

		def token_total(field: str) -> int | None:
			if not llm_events:
				return 0 if instrumentation_available else None
			values = [event.get(field) for event in llm_events]
			if any(value is None for value in values):
				return None
			return sum(value for value in values if value is not None)

		total_event = next(
			(
				event
				for event in timing_events
				if event.get("stage") == "TOTAL_AGENT"
			),
			None,
		)
		latency_ms = (
			total_event.get("duration_ms")
			if total_event is not None
			else report_latencies.get(golden.case_id)
		)
		results.append(
			BenchmarkResult(
				case_id=golden.case_id,
				passed=passed,
				comparisons=comparisons,
				performance=CasePerformance(
					latency_ms=latency_ms,
					llm_call_count=(
						len(llm_events)
						if instrumentation_available else None
					),
					failed_llm_call_count=(
						sum(event.get("success") is False for event in llm_events)
						if instrumentation_available else None
					),
					judge_call_count=sum(
						event.get("stage", "").startswith("LLM_JUDGE:")
						for event in timing_events
					) if instrumentation_available else None,
					llm_retry_count=(
						trace.get("llm_retry_count")
						if isinstance(trace.get("llm_retry_count"), int)
						else (
							sum(
								event.get("stage", "").startswith("LLM_RETRY:")
								for event in timing_events
							)
							if any(
								event.get("stage", "").startswith("LLM_RETRY:")
								for event in timing_events
							)
							else None
						)
					),
					retry_count=(
						sum(trace.get("tool_retry_count", {}).values())
						if isinstance(trace.get("tool_retry_count"), dict)
						else None
					),
					prompt_token_count=token_total("prompt_tokens"),
					completion_token_count=token_total("completion_tokens"),
					total_token_count=token_total("total_tokens"),
				),
				execution_status=(
					"failed"
					if total_event is not None
					and total_event.get("success") is False
					else "completed"
				),
				category=golden.category,
				customer_message=golden.customer_message,
				actual_response=trace.get("response"),
			)
		)
	return results


def _approved(
	values: dict[str, Any],
	*,
	actual: bool,
) -> bool | None:
	return automatic_refund_label(
		values.get("settlement.decision"),
		values.get("settlement.refund_amount"),
		values.get("escalation"),
	)


def _actual_escalation(values: dict[str, Any]) -> bool | None:
	if values.get("settlement.decision") == "ESCALATE":
		return True
	escalation = values.get("escalation")
	return escalation if isinstance(escalation, bool) else None


def _case_artifact(result: BenchmarkResult) -> dict[str, Any]:
	expected = _expected_values(result)
	actual = _actual_values(result)
	comparisons = {item.field: item for item in result.comparisons}
	required = (
		"settlement.decision",
		"settlement.refund_amount",
		"escalation",
		"intent",
		"grievance_types",
		"judgements",
	)
	csat = None
	if (
		result.actual_response
		and result.actual_response.strip()
		and all(field in comparisons for field in required)
		and all(comparisons[field].actual is not None for field in required)
	):
		correctness = all(
			comparisons[field].passed
			for field in (
				"settlement.decision",
				"settlement.refund_amount",
				"escalation",
			)
		)
		fairness = all(
			comparisons[field].passed
			for field in ("settlement.decision", "escalation")
		)
		completeness = all(
			comparisons[field].passed
			for field in ("intent", "grievance_types", "judgements")
		)
		clarity = 5 if 20 <= len(result.actual_response.strip()) <= 2000 else 2
		csat = round(
			(
				(5 if correctness else 1)
				+ (5 if fairness else 1)
				+ (5 if completeness else 1)
				+ clarity
			)
			/ 4,
			1,
		)
	performance = result.performance
	latency_ms = (
		performance.latency_ms
		if performance is not None
		and performance.latency_ms is not None
		and isfinite(performance.latency_ms)
		and performance.latency_ms >= 0
		else None
	)
	return {
		"case_id": result.case_id,
		"category": result.category,
		"customer_issue": result.customer_message,
		"actual_response": result.actual_response,
		"expected": expected,
		"actual": actual,
		"passed": result.passed,
		"execution_status": result.execution_status,
		"execution_error": result.execution_error,
		"escalated": _actual_escalation(actual),
		"automated": _approved(actual, actual=True) is True,
		"expected_refund_approved": _approved(expected, actual=False),
		"actual_refund_approved": _approved(actual, actual=True),
		"latency_ms": latency_ms,
		"llm_call_count": performance.llm_call_count if performance else None,
		"failed_llm_call_count": (
			performance.failed_llm_call_count if performance else None
		),
		"judge_call_count": performance.judge_call_count if performance else None,
		"llm_retry_count": performance.llm_retry_count if performance else None,
		"tool_retry_count": performance.retry_count if performance else None,
		"prompt_tokens": performance.prompt_token_count if performance else None,
		"completion_tokens": (
			performance.completion_token_count if performance else None
		),
		"total_tokens": (
			performance.total_token_count if performance else None
		),
		"simulated_csat": csat,
		"display_actual_decision": (
			actual.get("settlement.decision") or "Not available"
		),
		"display_expected_outcome": (
			f"{expected.get('settlement.decision') or 'Not available'} · "
			f"{expected.get('settlement.refund_amount', 'Not available')} refund · "
			f"human review: {expected.get('escalation', 'Not available')}"
		),
		"display_latency": (
			f"{performance.latency_ms:.0f} ms"
			if latency_ms is not None
			else "Not available"
		),
		"display_simulated_csat": (
			f"{csat:.1f} / 5" if csat is not None else "Not available"
		),
	}


def build_dashboard_artifact(
	results: list[BenchmarkResult],
	*,
	review_minutes_per_escalation: float = 5,
	input_cost_per_million_tokens: float | None = None,
	output_cost_per_million_tokens: float | None = None,
	token_pricing_model: str | None = None,
	token_pricing_provider: str | None = None,
	token_pricing_endpoint: str | None = None,
	token_pricing_source: str | None = None,
	cost_currency: str = "USD",
	manual_handling_cost_per_case: float | None = 8,
	human_review_cost_per_hour: float | None = 30,
	manual_resolution_minutes_per_case: float = 12,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	if (
		manual_handling_cost_per_case is not None
		and manual_handling_cost_per_case < 0
	) or (
		human_review_cost_per_hour is not None
		and human_review_cost_per_hour < 0
	):
		raise ValueError("Business cost assumptions cannot be negative.")
	if manual_resolution_minutes_per_case < 0:
		raise ValueError("Manual resolution minutes cannot be negative.")

	cases = [_case_artifact(result) for result in results]
	business = calculate_business_impact_metrics(
		results,
		review_minutes_per_escalation=review_minutes_per_escalation,
		input_cost_per_million_tokens=input_cost_per_million_tokens,
		output_cost_per_million_tokens=output_cost_per_million_tokens,
		manual_handling_cost_per_case=manual_handling_cost_per_case,
		human_review_cost_per_hour=human_review_cost_per_hour,
	)
	quality = calculate_metrics(results)
	performance = calculate_performance_metrics(results)
	decision_metrics = calculate_decision_metrics(results)
	confusion = decision_metrics["decision_confusion_matrix"]

	escalations = [case for case in cases if case["escalated"] is True]
	reason_counts = Counter(
		reason
		for case in escalations
		for reason in (case["actual"].get("grievance_types") or [])
	)
	latencies = sorted(
		case["latency_ms"]
		for case in cases
		if case["latency_ms"] is not None
	)
	csat_scores = [
		case["simulated_csat"]
		for case in cases
		if case["simulated_csat"] is not None
	]
	csat_distribution = [
		{"rating": str(rating), "cases": count}
		for rating, count in sorted(Counter(csat_scores).items())
	]
	latency_buckets = Counter(
		int(case["latency_ms"] // 1000)
		for case in cases
		if case["latency_ms"] is not None
	)
	confusion_rows = None
	if decision_metrics["decision_confusion_matrix_denominator"] > 0:
		confusion_rows = [
			{
				"Actual / expected": "Refund approved",
				"Expected approved": confusion["true_positive"],
				"Expected not approved": confusion["false_positive"],
			},
			{
				"Actual / expected": "Not approved",
				"Expected approved": confusion["false_negative"],
				"Expected not approved": confusion["true_negative"],
			},
		]

	recommendations = []
	for key, priority, title in (
		("false_positive", "High", "Review incorrect automatic refunds"),
		("false_negative", "High", "Investigate missed eligible refunds"),
	):
		count = confusion[key]
		if count:
			case_ids = [
				case["case_id"]
				for case in cases
				if (
					case["actual_refund_approved"] is True
					and case["expected_refund_approved"] is False
					if key == "false_positive"
					else case["actual_refund_approved"] is False
					and case["expected_refund_approved"] is True
				)
			]
			recommendations.append(
				{
					"priority": priority,
					"title": title,
					"explanation": f"{count} case(s) need decision review.",
					"case_ids": case_ids,
				}
			)
	if escalations:
		recommendations.append(
			{
				"priority": "Medium",
				"title": "Investigate human-review drivers",
				"explanation": (
					f"{len(escalations)} case(s) required human review. "
					"Review the most common recorded grievance reasons."
				),
				"case_ids": [case["case_id"] for case in escalations],
			}
		)
	slow_cases = [
		case["case_id"]
		for case in cases
		if performance.p95_latency_ms is not None
		and case["latency_ms"] is not None
		and case["latency_ms"] > performance.p95_latency_ms
	]
	if slow_cases:
		recommendations.append(
			{
				"priority": "Low",
				"title": "Review slower resolutions",
				"explanation": "These cases are at the slow end of measured processing time.",
				"case_ids": slow_cases,
			}
		)
	if not recommendations:
		recommendations.append(
			{
				"priority": "Low",
				"title": "No priority issue identified",
				"explanation": (
					"No incorrect refund decisions or escalations were observed "
					"among classified benchmark outcomes."
				),
				"case_ids": [],
			}
		)

	currency = cost_currency.upper()
	dashboard = {
		"schema_version": 1,
		"case_count": len(results),
		"assumptions": {
			"review_minutes_per_escalation": review_minutes_per_escalation,
			"manual_handling_cost_per_case": manual_handling_cost_per_case,
			"human_review_cost_per_hour": human_review_cost_per_hour,
			"manual_resolution_minutes_per_case": manual_resolution_minutes_per_case,
			"input_cost_per_million_tokens": input_cost_per_million_tokens,
			"output_cost_per_million_tokens": output_cost_per_million_tokens,
			"token_pricing_model": token_pricing_model,
			"token_pricing_provider": token_pricing_provider,
			"token_pricing_endpoint": token_pricing_endpoint,
			"token_pricing_source": token_pricing_source,
			"cost_currency": currency,
			"latency_timing_boundary": (
				"TOTAL_AGENT measures the orchestrator run from run_agent entry "
				"through workflow completion, including its internal tools and "
				"decision work; it excludes surrounding application-service "
				"bookkeeping."
			),
			"business_impact_methodology": (
				"Estimated handling cost avoided counts only cases with an "
				"automatic approval and passing deterministic decision-field "
				"comparisons for the decision, refund amount, and escalation. "
				"Estimated net savings subtract estimated human-review labor and "
				"estimated LLM cost. Error costs, churn, infrastructure, and "
				"unmodeled support costs are excluded; this is not realized savings."
			),
			"simulated_csat_rubric": (
				"1-5 average of correctness, decision fairness, completeness, "
				"and response clarity proxy; scores are simulated."
			),
			"workflow_outcome_source": (
				"Recorded by the orchestrator as RESOLVED, ESCALATED, or "
				"DEFLECTED. Missing outcomes remain unavailable; automation "
				"and escalation flags are not used to fill them."
			),
		},
		"metrics": {
			"estimated_net_savings": business.estimated_net_savings,
			"estimated_savings_per_case": business.estimated_savings_per_case,
			"estimated_manual_handling_cost_avoided": (
				business.estimated_manual_handling_cost_avoided
			),
			"estimated_human_review_cost": business.estimated_human_review_cost,
			"savings_unavailable_reasons": business.savings_unavailable_reasons,
			"simulated_csat": (
				sum(csat_scores) / len(csat_scores) if csat_scores else None
			),
			"simulated_csat_scored_case_count": len(csat_scores),
			"simulated_csat_unscored_case_count": (
				len(cases) - len(csat_scores)
			),
			"simulated_csat_coverage": (
				len(csat_scores) / len(cases) if cases else None
			),
			"simulated_csat_is_actual_feedback": False,
			"incorrect_refund_decisions": {
				"false_positives": decision_metrics["false_positive_count"],
				"false_negatives": decision_metrics["false_negative_count"],
				"false_positive_rate": decision_metrics["false_positive_rate"],
				"false_negative_rate": decision_metrics["false_negative_rate"],
			},
			"decision_confusion_matrix": confusion,
			"decision_confusion_matrix_denominator": (
				decision_metrics["decision_confusion_matrix_denominator"]
			),
			"decision_confusion_matrix_evaluated_case_count": (
				decision_metrics[
					"decision_confusion_matrix_evaluated_case_count"
				]
			),
			"decision_confusion_matrix_excluded_case_count": (
				decision_metrics[
					"decision_confusion_matrix_excluded_case_count"
				]
			),
			"decision_confusion_matrix_excluded_expected_ambiguous_count": (
				decision_metrics[
					"decision_confusion_matrix_excluded_expected_ambiguous_count"
				]
			),
			"decision_confusion_matrix_excluded_actual_ambiguous_count": (
				decision_metrics[
					"decision_confusion_matrix_excluded_actual_ambiguous_count"
				]
			),
			"agent_escalation_count": decision_metrics["agent_escalation_count"],
			"agent_escalations_excluded_from_confusion_matrix": (
				decision_metrics[
					"agent_escalations_excluded_from_confusion_matrix"
				]
			),
			"agent_escalations_classified_as_not_automatically_approved": (
				decision_metrics[
					"agent_escalations_classified_as_not_automatically_approved"
				]
			),
			"precision": decision_metrics["precision"],
			"recall": decision_metrics["recall"],
			"specificity": decision_metrics["specificity"],
			"f1_score": decision_metrics["f1_score"],
			"false_positive_count": decision_metrics["false_positive_count"],
			"false_positive_rate": decision_metrics["false_positive_rate"],
			"false_negative_count": decision_metrics["false_negative_count"],
			"false_negative_rate": decision_metrics["false_negative_rate"],
			"decision_metrics": decision_metrics,
			"ambiguous_escalated_cases_excluded_from_confusion_matrix": 0,
			"ambiguous_cases_excluded_from_confusion_matrix": (
				decision_metrics["decision_confusion_matrix_excluded_case_count"]
			),
			"confusion_matrix_rows": confusion_rows,
			"automation_rate": business.automation_rate,
			"automation_rate_denominator": (
				business.automation_rate_denominator
			),
			"escalation_rate": business.escalation_rate,
			"escalation_rate_denominator": (
				business.escalation_rate_denominator
			),
			"escalation_count": business.escalation_count,
			"case_count": business.case_count,
			"automated_count": business.automated_count,
			"successful_automated_resolution_count": (
				business.successful_automated_resolution_count
			),
			"other_outcome_count": business.other_outcome_count,
			"outcome_unavailable_count": business.outcome_unavailable_count,
			"outcome_classified_case_count": (
				business.outcome_classified_case_count
			),
			"workflow_outcome_classified_case_count": (
				business.workflow_outcome_classified_case_count
			),
			"workflow_outcome_unavailable_count": (
				business.workflow_outcome_unavailable_count
			),
			"workflow_outcome_distribution": (
				business.workflow_outcome_distribution
			),
			"workflow_escalation_count": business.workflow_escalation_count,
			"estimated_human_review_minutes": business.estimated_human_review_minutes,
			"total_input_tokens": business.total_input_tokens,
			"total_output_tokens": business.total_output_tokens,
			"estimated_llm_cost": business.estimated_llm_cost,
			"estimated_input_cost": business.estimated_input_cost,
			"estimated_output_cost": business.estimated_output_cost,
			"average_latency_ms": performance.average_latency_ms,
			"p50_latency_ms": performance.p50_latency_ms,
			"p95_latency_ms": performance.p95_latency_ms,
			"latency_measured_case_count": (
				performance.latency_measured_case_count
			),
			"latency_missing_case_count": (
				performance.latency_missing_case_count
			),
			"latency_coverage": performance.latency_coverage,
			"latency_case_count": performance.case_count,
			"latencies_ms": [case["latency_ms"] for case in cases],
			"escalated_count": len(escalations),
			"csat_distribution": csat_distribution,
			"outcome_by_case": [
				{
					"case_id": case["case_id"],
					"automated": int(case["automated"]),
					"escalated": int(case["escalated"] is True),
				}
				for case in cases
			],
			"latency_distribution": [
				{
					"duration": f"{second}–{second + 1} s",
					"cases": count,
				}
				for second, count in sorted(latency_buckets.items())
			],
			"top_escalation_reasons": [
				{"reason": reason, "count": count}
				for reason, count in reason_counts.most_common()
			],
			"quality_pass_rate": quality.pass_rate,
			"completed_case_count": quality.completed_case_count,
			"execution_failure_count": quality.execution_failure_count,
			"field_evaluation_coverage": quality.field_evaluation_coverage,
			"performance": performance.model_dump(),
			"llm_telemetry_coverage": {
				"instrumented_case_count": performance.instrumented_case_count,
				"uninstrumented_case_count": (
					performance.uninstrumented_case_count
				),
			},
			"llm_call_count": performance.llm_call_count,
			"failed_llm_call_count": performance.failed_llm_call_count,
			"judge_call_count": performance.judge_call_count,
			"llm_retry_count": performance.llm_retry_count,
			"tool_retry_count": performance.tool_retry_count,
			"total_retry_count": performance.total_retry_count,
			"execution_failure_count": performance.execution_failure_count,
			"business_impact": business.model_dump(),
			"manual_resolution_minutes_assumption": manual_resolution_minutes_per_case,
		},
		"recommendations": recommendations,
		"historical_trend": None,
		"trend_note": "Single benchmark run; no historical trend is available.",
	}
	return dashboard, cases


def write_dashboard_artifacts(
	results: list[BenchmarkResult],
	output_dir: Path,
	**assumptions: Any,
) -> tuple[Path, Path]:
	dashboard, cases = build_dashboard_artifact(results, **assumptions)
	output_dir.mkdir(parents=True, exist_ok=True)
	case_path = output_dir / "case_results.json"
	dashboard_path = output_dir / "dashboard.json"
	case_path.write_text(
		json.dumps(cases, indent=2, ensure_ascii=False) + "\n",
		encoding="utf-8",
	)
	dashboard_path.write_text(
		json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n",
		encoding="utf-8",
	)
	return case_path, dashboard_path
