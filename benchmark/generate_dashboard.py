import argparse
import json
import os
import re
from pathlib import Path

from benchmark.dashboard_artifacts import (
	load_saved_evaluation_results,
	write_dashboard_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "output.txt"
DEFAULT_GOLDEN_SET = PROJECT_ROOT / "data" / "golden_set.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmark_results"
PRICED_MODEL = "gpt-5.6-luna"
PRICED_PROVIDER = "OPENAI"
PRICED_ENDPOINT = "https://api.openai.com/v1"
INPUT_PRICE_PER_MILLION = 0.20
OUTPUT_PRICE_PER_MILLION = 1.20


def _default_prices_for_saved_run(report: str) -> tuple[float | None, float | None]:
	starts = list(re.finditer(r'(?m)^\{\s*\n\s+"case_id":', report))
	decoder = json.JSONDecoder()
	traces = [decoder.raw_decode(report, match.start())[0] for match in starts]
	llm_events = [
		event
		for trace in traces
		for event in trace.get("timing_events", [])
		if event.get("model") is not None
	]
	if llm_events and all(
		event.get("model") == PRICED_MODEL
		and event.get("provider") == PRICED_PROVIDER
		and event.get("endpoint", "").rstrip("/") == PRICED_ENDPOINT
		and isinstance(event.get("prompt_tokens"), int)
		and event["prompt_tokens"] <= 272_000
		for event in llm_events
	):
		return INPUT_PRICE_PER_MILLION, OUTPUT_PRICE_PER_MILLION
	return None, None


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Build dashboard JSON from a previously saved benchmark report. "
			"This command does not run evaluations or call an LLM."
		)
	)
	parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
	parser.add_argument(
		"--golden-set",
		type=Path,
		default=DEFAULT_GOLDEN_SET,
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
	)
	parser.add_argument("--review-minutes-per-escalation", type=float, default=5)
	parser.add_argument("--cost-currency", default=os.getenv("REFUNDBENCH_COST_CURRENCY", "USD"))
	parser.add_argument(
		"--input-cost-per-million-tokens",
		type=float,
		default=(
			float(os.environ["REFUNDBENCH_INPUT_COST_PER_MILLION_TOKENS"])
			if os.getenv("REFUNDBENCH_INPUT_COST_PER_MILLION_TOKENS")
			else None
		),
	)
	parser.add_argument(
		"--output-cost-per-million-tokens",
		type=float,
		default=(
			float(os.environ["REFUNDBENCH_OUTPUT_COST_PER_MILLION_TOKENS"])
			if os.getenv("REFUNDBENCH_OUTPUT_COST_PER_MILLION_TOKENS")
			else None
		),
	)
	parser.add_argument("--manual-handling-cost-per-case", type=float, default=8)
	parser.add_argument("--human-review-cost-per-hour", type=float, default=30)
	parser.add_argument("--manual-resolution-minutes-per-case", type=float, default=12)
	args = parser.parse_args()

	report_path = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
	golden_path = (
		args.golden_set
		if args.golden_set.is_absolute()
		else PROJECT_ROOT / args.golden_set
	)
	output_dir = (
		args.output_dir
		if args.output_dir.is_absolute()
		else PROJECT_ROOT / args.output_dir
	)
	try:
		report_text = report_path.read_text(encoding="utf-8")
		results = load_saved_evaluation_results(report_path, golden_path)
	except (OSError, json.JSONDecodeError, ValueError) as error:
		parser.error(f"Cannot build dashboard from saved evaluation: {error}")

	default_input, default_output = _default_prices_for_saved_run(report_text)
	input_price = (
		args.input_cost_per_million_tokens
		if args.input_cost_per_million_tokens is not None
		else default_input
	)
	output_price = (
		args.output_cost_per_million_tokens
		if args.output_cost_per_million_tokens is not None
		else default_output
	)
	case_path, dashboard_path = write_dashboard_artifacts(
		results,
		output_dir,
		review_minutes_per_escalation=args.review_minutes_per_escalation,
		input_cost_per_million_tokens=input_price,
		output_cost_per_million_tokens=output_price,
		token_pricing_model=(
			PRICED_MODEL if default_input is not None else None
		),
		token_pricing_provider=(
			PRICED_PROVIDER if default_input is not None else None
		),
		token_pricing_endpoint=(
			PRICED_ENDPOINT if default_input is not None else None
		),
		token_pricing_source=(
			"saved model, provider, endpoint, and context metadata"
			if default_input is not None
			else (
				"explicit token price configuration"
				if args.input_cost_per_million_tokens is not None
				or args.output_cost_per_million_tokens is not None
				else "not configured"
			)
		),
		cost_currency=args.cost_currency,
		manual_handling_cost_per_case=args.manual_handling_cost_per_case,
		human_review_cost_per_hour=args.human_review_cost_per_hour,
		manual_resolution_minutes_per_case=(
			args.manual_resolution_minutes_per_case
		),
	)
	print(f"Loaded {len(results)} cases from {report_path}")
	print(f"Wrote case results: {case_path}")
	print(f"Wrote dashboard: {dashboard_path}")


if __name__ == "__main__":
	main()
