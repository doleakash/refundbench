import argparse
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from app.agent.state import AgentState
from app.application.cases.service import CaseApplicationService
from app.bootstrap import build_container
from app.infrastructure.observability import capture_timing_events
from app.policy.engine import PolicyAction
from app.settlement.refund_ledger import RefundRecord
from benchmark.comparator import compare_outcomes
from benchmark.dashboard_artifacts import write_dashboard_artifacts
from benchmark.metrics import (
	calculate_business_impact_metrics,
	calculate_decision_metrics,
	calculate_metrics,
	calculate_performance_metrics,
)
from benchmark.models import (
	ActualGrievance,
	ActualJudgement,
	ActualOutcome,
	ActualSettlement,
	BenchmarkResult,
	CasePerformance,
	GoldenCase,
)
from config.settings import get_settings

PRICED_OPENAI_MODEL = "gpt-5.6-luna"
PRICED_OPENAI_INPUT_COST_PER_MILLION_TOKENS = 0.20
PRICED_OPENAI_OUTPUT_COST_PER_MILLION_TOKENS = 1.20
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SET_PATH = (
	PROJECT_ROOT / "data" / "golden_set.json"
)
DEFAULT_DASHBOARD_ARTIFACT_DIR = (
	PROJECT_ROOT / "data" / "benchmark_results"
)
logger = logging.getLogger(__name__)


def resolve_project_path(path: Path) -> Path:
	"""Resolve relative CLI paths from the repository root."""
	return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load_golden_cases(path: Path = GOLDEN_SET_PATH) -> list[GoldenCase]:
	with path.open(encoding="utf-8") as golden_set_file:
		data = json.load(golden_set_file)

	raw_cases = data if isinstance(data, list) else [data]
	return [GoldenCase.model_validate(case) for case in raw_cases]


def _actual_outcome(
	state: AgentState,
	case_service: CaseApplicationService,
) -> ActualOutcome:
	settlement = state.settlement
	refund = case_service.refund_workflow.settlement.ledger.get_refund(
		state.order_id
	)

	return ActualOutcome(
		intent=state.intent.value,
		outcome=state.outcome,
		grievances=[
			ActualGrievance(type=grievance.type.value)
			for grievance in state.grievances
		],
		judgements=[
			ActualJudgement(
				grievance_type=grievance.type.value,
				verdict=state.consensus[grievance.grievance_id],
			)
			for grievance in state.grievances
			if grievance.grievance_id in state.consensus
		],
		settlement=(
			ActualSettlement(
				decision=(
					settlement.status.value
					if settlement is not None
					else "ESCALATE"
				),
				refund_amount=(
					settlement.total_refund
					if settlement is not None
					else 0.0
				),
				refund_status=refund.status if refund is not None else None,
				idempotency_key=(
					refund.idempotency_key
					if refund is not None
					else None
				),
			)
			if settlement is not None
			   or (state.actions and state.actions[-1] == "ESCALATE")
			else None
		),
		escalation=(
			bool(state.escalation_reason)
			or (
				bool(state.actions)
				and state.actions[-1] == "ESCALATE"
			)
			or any(
			decision.action is PolicyAction.ESCALATE
			for decision in state.policy_decisions.values()
		)
			or (
				settlement is not None
				and settlement.status.value == "ESCALATE"
			)
		),
	)


def _apply_preconditions(
	golden_case: GoldenCase,
	case_service: CaseApplicationService,
) -> None:
	if golden_case.preconditions is None:
		return

	existing_refund = golden_case.preconditions.existing_refund
	if existing_refund is None:
		return

	ledger = case_service.refund_workflow.settlement.ledger

	refund: RefundRecord = ledger.create_refund(
		case_id=golden_case.case_id,
		order_id=golden_case.order_id,
		amount=existing_refund.amount,
	)

	if refund.amount != existing_refund.amount:
		raise ValueError(
			f"Existing refund amount cannot be seeded for order "
			f"{golden_case.order_id}."
		)

	if refund.idempotency_key != existing_refund.idempotency_key:
		raise ValueError(
			f"Existing refund idempotency key cannot be seeded for order "
			f"{golden_case.order_id}."
		)

	if existing_refund.status == "PROPOSED":
		pass
	elif existing_refund.status in {"PROCESSING", "REFUNDED"}:
		ledger.accept_refund(golden_case.order_id)

		if existing_refund.status == "REFUNDED":
			ledger.mark_refunded(golden_case.order_id)
	else:
		raise ValueError(
			f"Existing refund status cannot be seeded using the refund "
			f"lifecycle API: {existing_refund.status}"
		)

	if refund.status != existing_refund.status:
		raise ValueError(
			f"Existing refund status {existing_refund.status} cannot be "
			f"seeded for order {golden_case.order_id}."
		)


def _llm_call_events(timing_events: list) -> list:
	return [
		event
		for event in timing_events
		if event.model is not None
		   and not event.stage.startswith("LLM_RETRY:")
		   and not (
			event.stage.startswith("LLM_JUDGE:")
			and event.success is False
			and getattr(event, "provider", None) is None
		)
	]


def _run_case(
	golden_case: GoldenCase,
	service: CaseApplicationService,
) -> BenchmarkResult:
	_apply_preconditions(golden_case, service)

	with capture_timing_events(golden_case.case_id) as captured_events:
		try:
			state = service.process_case(
				order_id=golden_case.order_id,
				customer_message=golden_case.customer_message,
			)
		except Exception as error:
			logger.exception(
				"Benchmark execution failed for case %s",
				golden_case.case_id,
			)
			timing_events = captured_events
			total_event = next(
				(
					event for event in timing_events
					if event.stage == "TOTAL_AGENT"
				),
				None,
			)
			llm_events = _llm_call_events(timing_events)
			instrumentation_available = bool(timing_events)

			def failed_token_total(field: str) -> int | None:
				values = [getattr(event, field) for event in llm_events]
				if not values:
					return 0 if instrumentation_available else None
				if any(value is None for value in values):
					return None
				return sum(values)

			return BenchmarkResult(
				case_id=golden_case.case_id,
				passed=False,
				comparisons=[],
				performance=CasePerformance(
					latency_ms=(
						total_event.duration_ms
						if total_event is not None
						else None
					),
					llm_call_count=(
						len(llm_events)
						if instrumentation_available
						else None
					),
					failed_llm_call_count=(
						sum(not event.success for event in llm_events)
						if instrumentation_available
						else None
					),
					judge_call_count=(
						sum(
							event.stage.startswith("LLM_JUDGE:")
							for event in timing_events
						)
						if instrumentation_available
						else None
					),
					llm_retry_count=(
						sum(
							event.stage.startswith("LLM_RETRY:")
							for event in timing_events
						)
						if instrumentation_available
						else None
					),
					prompt_token_count=failed_token_total("prompt_tokens"),
					completion_token_count=failed_token_total(
						"completion_tokens"
					),
					total_token_count=failed_token_total("total_tokens"),
				),
				execution_status="failed",
				execution_error=f"{type(error).__name__}: {error}",
				category=golden_case.category,
				customer_message=golden_case.customer_message,
			)

	result = compare_outcomes(
		case_id=golden_case.case_id,
		expected=golden_case.expected,
		actual=_actual_outcome(state, service),
	)
	timing_events = captured_events or state.timing_events
	total_event = next(
		(
			event
			for event in timing_events
			if event.stage == "TOTAL_AGENT"
		),
		None,
	)
	llm_events = _llm_call_events(timing_events)

	instrumentation_available = total_event is not None or bool(timing_events)

	def total_tokens(values: list[int | None]) -> int | None:
		if not values:
			return 0 if instrumentation_available else None
		if any(value is None for value in values):
			return None
		return sum(value for value in values if value is not None)

	llm_retry_count = sum(
		event.stage.startswith("LLM_RETRY:")
		for event in timing_events
	)
	result.performance = CasePerformance(
		latency_ms=(
			total_event.duration_ms if total_event is not None else None
		),
		llm_call_count=(
			len(llm_events) if instrumentation_available else None
		),
		failed_llm_call_count=(
			sum(not event.success for event in llm_events)
			if instrumentation_available
			else None
		),
		judge_call_count=sum(
			event.stage.startswith("LLM_JUDGE:")
			for event in timing_events
		) if instrumentation_available else None,
		llm_retry_count=(
			llm_retry_count if instrumentation_available else None
		),
		retry_count=(
			sum(state.tool_retry_count.values())
			if instrumentation_available
			else None
		),
		prompt_token_count=total_tokens(
			[event.prompt_tokens for event in llm_events]
		),
		completion_token_count=total_tokens(
			[event.completion_tokens for event in llm_events]
		),
		total_token_count=total_tokens(
			[event.total_tokens for event in llm_events]
		),
	)
	if total_event is not None and not total_event.success:
		result.execution_status = "failed"
		result.execution_error = "TOTAL_AGENT timing event reported failure."
	result.category = getattr(golden_case, "category", None)
	result.customer_message = golden_case.customer_message
	result.actual_response = getattr(state, "response", None)
	return result


def run_benchmark(
	golden_set_path: Path = GOLDEN_SET_PATH,
	case_service: CaseApplicationService | None = None,
	case_id: str | None = None,
) -> list[BenchmarkResult]:
	golden_cases = load_golden_cases(golden_set_path)

	if case_id is not None:
		golden_cases = [
			case for case in golden_cases if case.case_id == case_id
		]

		if not golden_cases:
			raise ValueError(f"Golden case not found: {case_id}")

	results: list[BenchmarkResult] = []

	for golden_case in golden_cases:
		try:
			if case_service is not None:
				results.append(_run_case(golden_case, case_service))
			else:
				with TemporaryDirectory(
					prefix="refundbench-"
				) as ledger_dir:
					service = build_container(
						ledger_data_dir=Path(ledger_dir),
					).case_service

					results.append(
						_run_case(golden_case, service)
					)
		except Exception as error:
			logger.exception(
				"Benchmark execution failed for case %s",
				golden_case.case_id,
			)
			results.append(
				BenchmarkResult(
					case_id=golden_case.case_id,
					passed=False,
					comparisons=[],
					execution_status="failed",
					execution_error=(
						f"{type(error).__name__}: {error}"
					),
					category=golden_case.category,
					customer_message=golden_case.customer_message,
				)
			)

	return results


def main() -> None:
	settings = get_settings()
	openai_model_pricing_matches = (
		settings.llm_provider == "OPENAI"
		and settings.open_ai_model == PRICED_OPENAI_MODEL
		and settings.llm_base_url in {
			None,
			"https://api.openai.com/v1",
		}
	)
	default_input_price = (
		os.getenv("REFUNDBENCH_INPUT_COST_PER_MILLION_TOKENS")
		or (
			str(PRICED_OPENAI_INPUT_COST_PER_MILLION_TOKENS)
			if openai_model_pricing_matches
			else None
		)
	)
	default_output_price = (
		os.getenv("REFUNDBENCH_OUTPUT_COST_PER_MILLION_TOKENS")
		or (
			str(PRICED_OPENAI_OUTPUT_COST_PER_MILLION_TOKENS)
			if openai_model_pricing_matches
			else None
		)
	)

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--case",
		help="Run only the golden case with this case ID.",
	)
	parser.add_argument(
		"--review-minutes-per-escalation",
		type=float,
		default=5,
		help="Estimated human-review minutes per escalated case (default: 5).",
	)
	parser.add_argument(
		"--input-cost-per-million-tokens",
		type=float,
		default=default_input_price,
		help=(
			"Input token price per million tokens, in the currency "
			"selected by --cost-currency. Also configurable with "
			"REFUNDBENCH_INPUT_COST_PER_MILLION_TOKENS; defaults to "
			"USD 0.20 for configured OpenAI gpt-5.6-luna standard "
			"short-context pricing (up to 272K input tokens per request)."
		),
	)
	parser.add_argument(
		"--output-cost-per-million-tokens",
		type=float,
		default=default_output_price,
		help=(
			"Output token price per million tokens, in the currency "
			"selected by --cost-currency. Also configurable with "
			"REFUNDBENCH_OUTPUT_COST_PER_MILLION_TOKENS; defaults to "
			"USD 1.20 for configured OpenAI gpt-5.6-luna standard "
			"short-context pricing (up to 272K input tokens per request)."
		),
	)
	parser.add_argument(
		"--cost-currency",
		default=os.getenv("REFUNDBENCH_COST_CURRENCY", "USD"),
		help=(
			"Currency label for token prices and estimated cost "
			"(default: USD; also configurable with REFUNDBENCH_COST_CURRENCY)."
		),
	)
	parser.add_argument(
		"--manual-handling-cost-per-case",
		type=float,
		default=float(os.getenv("REFUNDBENCH_MANUAL_HANDLING_COST_PER_CASE", "8")),
		help="Assumed manual handling cost per case.",
	)
	parser.add_argument(
		"--human-review-cost-per-hour",
		type=float,
		default=float(os.getenv("REFUNDBENCH_HUMAN_REVIEW_COST_PER_HOUR", "30")),
		help="Assumed human-review labor cost per hour.",
	)
	parser.add_argument(
		"--manual-resolution-minutes-per-case",
		type=float,
		default=float(os.getenv("REFUNDBENCH_MANUAL_RESOLUTION_MINUTES_PER_CASE", "12")),
		help=(
			"Assumed manual resolution time per case, shown separately "
			"from measured agent latency."
		),
	)
	parser.add_argument(
		"--dashboard-artifacts-dir",
		type=Path,
		default=Path(
			os.getenv(
				"REFUNDBENCH_DASHBOARD_ARTIFACTS_DIR",
				str(DEFAULT_DASHBOARD_ARTIFACT_DIR),
			)
		),
		help="Directory for generated case_results.json and dashboard.json artifacts.",
	)
	args = parser.parse_args()
	args.dashboard_artifacts_dir = resolve_project_path(
		args.dashboard_artifacts_dir
	)

	try:
		results = run_benchmark(case_id=args.case)
	except ValueError as error:
		parser.error(str(error))

	if args.case is None:
		configured_endpoint = (
			settings.llm_base_url
			or "https://api.openai.com/v1"
		)
		pricing_source = (
			"configured model and endpoint default"
			if openai_model_pricing_matches
			else (
				"explicit token price configuration"
				if args.input_cost_per_million_tokens is not None
				   or args.output_cost_per_million_tokens is not None
				else "not configured"
			)
		)
		case_artifact_path, dashboard_artifact_path = write_dashboard_artifacts(
			results,
			args.dashboard_artifacts_dir,
			review_minutes_per_escalation=(
				args.review_minutes_per_escalation
			),
			input_cost_per_million_tokens=(
				args.input_cost_per_million_tokens
			),
			output_cost_per_million_tokens=(
				args.output_cost_per_million_tokens
			),
			token_pricing_model=settings.open_ai_model,
			token_pricing_provider=settings.llm_provider,
			token_pricing_endpoint=configured_endpoint,
			token_pricing_source=pricing_source,
			cost_currency=args.cost_currency,
			manual_handling_cost_per_case=(
				args.manual_handling_cost_per_case
			),
			human_review_cost_per_hour=args.human_review_cost_per_hour,
			manual_resolution_minutes_per_case=(
				args.manual_resolution_minutes_per_case
			),
		)
		print(
			"Dashboard artifacts: "
			f"{case_artifact_path} and {dashboard_artifact_path}"
		)

	for result in results:
		print(
			f"{result.case_id}  "
			f"{'PASS' if result.passed else 'FAIL'}"
		)

		if not result.passed:
			for comparison in result.comparisons:
				if not comparison.passed:
					print(
						f"  MISMATCH | {comparison.field} | "
						f"expected={comparison.expected!r} | "
						f"actual={comparison.actual!r}"
					)

	metrics = calculate_metrics(results)
	performance = calculate_performance_metrics(results)
	business_impact = calculate_business_impact_metrics(
		results,
		review_minutes_per_escalation=(
			args.review_minutes_per_escalation
		),
		input_cost_per_million_tokens=(
			args.input_cost_per_million_tokens
		),
		output_cost_per_million_tokens=(
			args.output_cost_per_million_tokens
		),
		manual_handling_cost_per_case=args.manual_handling_cost_per_case,
		human_review_cost_per_hour=args.human_review_cost_per_hour,
	)

	def format_percentage(value: float | None) -> str:
		return f"{value:.0%}" if value is not None else "N/A"

	def format_average(value: float | None) -> str:
		return f"{value:.2f}" if value is not None else "N/A"

	print()
	print(f"Cases:                  {len(results)}")
	print(f"Passed:                 {sum(result.passed for result in results)}")
	print(f"Failed:                 {sum(not result.passed for result in results)}")
	print()
	print(f"Pass Rate:              {format_percentage(metrics.pass_rate)}")
	print(f"Intent Accuracy:        {format_percentage(metrics.intent_accuracy)}")
	print(f"Grievance Accuracy:     {format_percentage(metrics.grievance_accuracy)}")
	print(f"Judgement Accuracy:     {format_percentage(metrics.judgement_accuracy)}")
	print(f"Refund Decision:        {format_percentage(metrics.refund_decision_accuracy)}")
	print(f"Refund Amount:          {format_percentage(metrics.refund_amount_accuracy)}")
	print(f"Escalation:             {format_percentage(metrics.escalation_accuracy)}")
	print(
		"Refund Status:          "
		+ (
			f"{metrics.refund_status_accuracy:.0%}"
			if metrics.refund_status_accuracy is not None
			else "N/A"
		)
	)
	print(
		"Idempotency Key:        "
		+ (
			f"{metrics.idempotency_key_accuracy:.0%}"
			if metrics.idempotency_key_accuracy is not None
			else "N/A"
		)
	)
	print()
	print("Performance")
	for case_id, latency_ms in performance.case_latencies_ms.items():
		print(
			f"  {case_id} end-to-end latency: "
			+ (f"{latency_ms:.2f} ms" if latency_ms is not None else "N/A")
		)
	print(
		f"  Latency coverage:      "
		f"{performance.latency_measured_case_count}/{len(results)} measured "
		f"({performance.latency_missing_case_count} missing)"
	)
	print(
		"  Average latency:       "
		+ (
			f"{performance.average_latency_ms:.2f} ms"
			if performance.average_latency_ms is not None
			else "N/A"
		)
	)
	print(
		"  P50 latency:           "
		+ (
			f"{performance.p50_latency_ms:.2f} ms"
			if performance.p50_latency_ms is not None
			else "N/A"
		)
	)
	print(
		"  P95 latency:           "
		+ (
			f"{performance.p95_latency_ms:.2f} ms"
			if performance.p95_latency_ms is not None
			else "N/A"
		)
	)
	print(
		f"  LLM telemetry coverage: "
		f"{performance.instrumented_case_count}/{len(results)} cases "
		f"({performance.uninstrumented_case_count} missing)"
	)
	print(
		f"  Average LLM calls:     "
		f"{format_average(performance.average_llm_call_count)}"
	)
	print(
		f"  Average judge calls:   "
		f"{format_average(performance.average_judge_call_count)}"
	)
	print(
		"  Total prompt tokens:   "
		+ (
			f"{performance.total_prompt_tokens:,} "
			f"({performance.prompt_token_case_count}/{len(results)} cases)"
			if performance.total_prompt_tokens is not None
			else "N/A (incomplete telemetry)"
		)
	)
	print(
		"  Total completion tokens: "
		+ (
			f"{performance.total_completion_tokens:,} "
			f"({performance.completion_token_case_count}/{len(results)} cases)"
			if performance.total_completion_tokens is not None
			else "N/A (incomplete telemetry)"
		)
	)
	print(
		"  Total tokens:          "
		+ (
			f"{performance.total_tokens:,} "
			f"({performance.total_token_case_count}/{len(results)} cases)"
			if performance.total_tokens is not None
			else "N/A (incomplete telemetry)"
		)
	)
	print(
		"  Failed LLM calls:      "
		+ (
			str(performance.failed_llm_call_count)
			if performance.failed_llm_call_count is not None
			else "N/A (incomplete instrumentation)"
		)
	)
	print(f"  LLM retries:           {performance.llm_retry_count}")
	print(f"  Tool retries:          {performance.tool_retry_count}")
	print(f"  Execution failures:    {performance.execution_failure_count}")
	print()
	print("Business Impact")
	print(
		f"  Automation rate:       "
		f"{format_percentage(business_impact.automation_rate)} "
		f"({business_impact.automation_rate_denominator} classifiable cases; "
		"automatic decisions, not successful-resolution rate)"
	)
	print(
		f"  Escalation rate:       "
		f"{format_percentage(business_impact.escalation_rate)} "
		f"({business_impact.escalation_rate_denominator} classifiable cases)"
	)
	print(f"  Escalation count:      {business_impact.escalation_count}")
	print(f"  Other outcomes:        {business_impact.other_outcome_count}")
	print(
		f"  Outcome unavailable:   "
		f"{business_impact.outcome_unavailable_count}"
	)
	print(
		f"  Correct auto resolutions: "
		f"{business_impact.successful_automated_resolution_count}"
	)
	print(
		f"  Manual handling cost assumption: "
		f"{args.cost_currency} "
		f"{args.manual_handling_cost_per_case:.2f}/case"
	)
	print(
		f"  Human-review cost assumption: "
		f"{args.cost_currency} "
		f"{args.human_review_cost_per_hour:.2f}/hour"
	)
	print(
		f"  Review-time assumption: "
		f"{args.review_minutes_per_escalation:.2f} minutes/escalation"
	)
	print(
		f"  Estimated review time: "
		+ (
			f"{business_impact.estimated_human_review_minutes:.2f} minutes"
			if business_impact.estimated_human_review_minutes is not None
			else "N/A (outcome telemetry incomplete)"
		)
	)
	print(
		f"  Estimated review cost: "
		+ (
			f"{args.cost_currency} "
			f"{business_impact.estimated_human_review_cost:.6f}"
			if business_impact.estimated_human_review_cost is not None
			else "N/A"
		)
	)
	print(
		f"  Estimated handling cost avoided: "
		+ (
			f"{args.cost_currency} "
			f"{business_impact.estimated_manual_handling_cost_avoided:.6f}"
			if business_impact.estimated_manual_handling_cost_avoided is not None
			else "N/A"
		)
	)
	print(
		"  Total benchmark input tokens: "
		+ (
			str(business_impact.total_input_tokens)
			if business_impact.total_input_tokens is not None
			else "N/A (incomplete token telemetry)"
		)
	)
	print(
		"  Total benchmark output tokens: "
		+ (
			str(business_impact.total_output_tokens)
			if business_impact.total_output_tokens is not None
			else "N/A (incomplete token telemetry)"
		)
	)
	print(
		"  Input price assumption: "
		+ (
			f"{args.cost_currency} "
			f"{args.input_cost_per_million_tokens:.6f} per million tokens"
			if args.input_cost_per_million_tokens is not None
			else "N/A (not configured)"
		)
	)
	print(
		"  Output price assumption: "
		+ (
			f"{args.cost_currency} "
			f"{args.output_cost_per_million_tokens:.6f} per million tokens"
			if args.output_cost_per_million_tokens is not None
			else "N/A (not configured)"
		)
	)
	print(
		"  Estimated input cost:  "
		+ (
			f"{args.cost_currency} "
			f"{business_impact.estimated_input_cost:.6f}"
			if business_impact.estimated_input_cost is not None
			else "N/A (requires input token data and input price)"
		)
	)
	print(
		"  Estimated output cost: "
		+ (
			f"{args.cost_currency} "
			f"{business_impact.estimated_output_cost:.6f}"
			if business_impact.estimated_output_cost is not None
			else "N/A (requires output token data and output price)"
		)
	)
	print(
		"  Estimated LLM cost:    "
		+ (
			f"{args.cost_currency} "
			f"{business_impact.estimated_llm_cost:.6f}"
			if business_impact.estimated_llm_cost is not None
			else "N/A (requires token data and both token prices)"
		)
	)
	print(
		"  Estimated cost per case: "
		+ (
			f"{args.cost_currency} "
			f"{business_impact.estimated_llm_cost_per_case:.6f}"
			if business_impact.estimated_llm_cost_per_case is not None
			else "N/A (requires token data and both token prices)"
		)
	)
	print(
		"  Estimated net savings: "
		+ (
			f"{args.cost_currency} "
			f"{business_impact.estimated_net_savings:.6f}"
			if business_impact.estimated_net_savings is not None
			else "N/A (requires complete outcomes, assumptions, token data, and prices)"
		)
	)
	print(
		"  Estimated savings per case: "
		+ (
			f"{args.cost_currency} "
			f"{business_impact.estimated_savings_per_case:.6f}"
			if business_impact.estimated_savings_per_case is not None
			else "N/A"
		)
	)
	print("  Savings limitation: excludes error, churn, infrastructure, and unmodeled support costs.")


if __name__ == "__main__":
	main()
