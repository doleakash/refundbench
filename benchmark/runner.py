import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from app.agent.state import AgentState
from app.application.cases.service import CaseApplicationService
from app.bootstrap import build_container
from app.policy.engine import PolicyAction
from app.settlement.refund_ledger import RefundRecord
from benchmark.comparator import compare_outcomes
from benchmark.metrics import (
	calculate_business_impact_metrics,
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
GOLDEN_SET_PATH = (
	Path(__file__).resolve().parent.parent
	/ "data"
	/ "golden_set.json"
)


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
		settlement=ActualSettlement(
			decision=(
				settlement.status.value
				if settlement is not None
				else (
					"ESCALATE"
					if state.actions
					   and state.actions[-1] == "ESCALATE"
					else None
				)
			),
			refund_amount=(
				settlement.total_refund
				if settlement is not None
				else 0.0
			),
			refund_status=(
				refund.status if refund is not None else None
			),
			idempotency_key=(
				refund.idempotency_key
				if refund is not None
				else None
			),
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


def _run_case(
	golden_case: GoldenCase,
	service: CaseApplicationService,
) -> BenchmarkResult:
	_apply_preconditions(golden_case, service)

	state = service.process_case(
		order_id=golden_case.order_id,
		customer_message=golden_case.customer_message,
	)

	result = compare_outcomes(
		case_id=golden_case.case_id,
		expected=golden_case.expected,
		actual=_actual_outcome(state, service),
	)
	total_event = next(
		(
			event
			for event in state.timing_events
			if event.stage == "TOTAL_AGENT"
		),
		None,
	)
	llm_events = [
		event
		for event in state.timing_events
		if event.model is not None
	]

	def total_tokens(values: list[int | None]) -> int | None:
		if not values:
			return 0 if state.timing_events else None
		if any(value is None for value in values):
			return None
		return sum(value for value in values if value is not None)

	result.performance = CasePerformance(
		latency_ms=(
			total_event.duration_ms if total_event is not None else None
		),
		llm_call_count=len(llm_events),
		judge_call_count=sum(
			event.stage.startswith("LLM_JUDGE:")
			for event in llm_events
		),
		retry_count=sum(state.tool_retry_count.values()),
		prompt_token_count=total_tokens(
			[event.prompt_tokens for event in llm_events]
		),
		completion_token_count=total_tokens(
			[event.completion_tokens for event in llm_events]
		),
		total_token_count=sum(
			event.total_tokens or 0 for event in llm_events
		),
	)
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
	args = parser.parse_args()

	try:
		results = run_benchmark(case_id=args.case)
	except ValueError as error:
		parser.error(str(error))

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
	)

	print()
	print(f"Cases:                  {len(results)}")
	print(f"Passed:                 {sum(result.passed for result in results)}")
	print(f"Failed:                 {sum(not result.passed for result in results)}")
	print()
	print(f"Pass Rate:              {metrics.pass_rate:.0%}")
	print(f"Intent Accuracy:        {metrics.intent_accuracy:.0%}")
	print(f"Grievance Accuracy:     {metrics.grievance_accuracy:.0%}")
	print(f"Judgement Accuracy:     {metrics.judgement_accuracy:.0%}")
	print(f"Refund Decision:        {metrics.refund_decision_accuracy:.0%}")
	print(f"Refund Amount:          {metrics.refund_amount_accuracy:.0%}")
	print(f"Escalation:             {metrics.escalation_accuracy:.0%}")
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
	print(f"  Average latency:       {performance.average_latency_ms:.2f} ms")
	print(f"  P50 latency:           {performance.p50_latency_ms:.2f} ms")
	print(f"  P95 latency:           {performance.p95_latency_ms:.2f} ms")
	print(
		f"  Average LLM calls:     "
		f"{performance.average_llm_call_count:.2f}"
	)
	print(
		f"  Average judge calls:   "
		f"{performance.average_judge_call_count:.2f}"
	)
	print(
		f"  Average prompt tokens: "
		f"{performance.average_prompt_token_count:.2f}"
	)
	print(
		f"  Average completion tokens: "
		f"{performance.average_completion_token_count:.2f}"
	)
	print(
		f"  Average total tokens:  "
		f"{performance.average_total_token_count:.2f}"
	)
	print(f"  Total retries:         {performance.total_retry_count}")
	print()
	print("Business Impact")
	print(f"  Automation rate:       {business_impact.automation_rate:.0%}")
	print(f"  Escalation rate:       {business_impact.escalation_rate:.0%}")
	print(f"  Escalation count:      {business_impact.escalation_count}")
	print(
		f"  Review-time assumption: "
		f"{args.review_minutes_per_escalation:.2f} minutes/escalation"
	)
	print(
		f"  Estimated review time: "
		f"{business_impact.estimated_human_review_minutes:.2f} minutes"
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
			else "N/A (requires token data and both token prices)"
		)
	)
	print(
		"  Estimated output cost: "
		+ (
			f"{args.cost_currency} "
			f"{business_impact.estimated_output_cost:.6f}"
			if business_impact.estimated_output_cost is not None
			else "N/A (requires token data and both token prices)"
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


if __name__ == "__main__":
	main()
