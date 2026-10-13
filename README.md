# RefundBench — Evidence-Grounded Refund Operations Agent

RefundBench is an evaluation-focused demo of an agent-assisted refund workflow. It combines tool-based order and delivery lookup, evidence-backed grievance judgments, explicit consensus, deterministic policy, and settlement tracking.

[Live dashboard](https://refundbench-dashboard.vercel.app/) · [Architecture PRD](https://refundbench-dashboard.vercel.app/)

## Problem

Refund decisions depend on several different signals: what the customer reports, what the order contains, what delivery records show, and what the refund policy permits. A useful system must keep those signals distinct, explain its decision path, and escalate when evidence or judge agreement is insufficient.

RefundBench demonstrates that workflow and evaluates it against a versioned golden-case dataset.

## Architecture

```text
Customer Request
       ↓
Streamlit UI → FastAPI → Case Service → Bounded Agent Orchestrator
                                        ├─ Order and delivery tools
                                        ├─ Grievance extraction
                                        ├─ Evidence construction
                                        └─ Independent LLM judges (parallel)
                                                   ↓
                                          2-of-3 Consensus
                                                   ↓
                                      Deterministic Refund Policy
                                                   ↓
                                      Conditional Settlement
                                                   ↓
                                 Response / Workflow Outcome

Golden Cases + Fixtures → Benchmark Runner → Comparator → Saved Run Artifacts
```

The runtime path and offline benchmark are separate. The benchmark compares actual case results with expected behavior; its metrics are not production telemetry.

### Decision pipeline

| Stage | Responsibility |
| --- | --- |
| Agent orchestrator | Selects and executes bounded workflow actions while maintaining case state. |
| Domain tools | Retrieve order and delivery records and construct grievance-specific evidence. |
| Judges | Assess claims independently using the available evidence. |
| Consensus | Returns a verdict when at least two judges agree; insufficient agreement escalates. |
| Policy | Applies deterministic refund rules to the evidence and consensus result. |
| Settlement | Records eligible settlement decisions and preserves refund idempotency by order. |
| Benchmark | Compares expected and actual case results and produces diagnostic metrics. |

Workflow outcomes (`RESOLVED`, `DEFLECTED`, `ESCALATED`) are distinct from settlement status and execution failures.

## Key engineering decisions

### Evidence before policy

Customer intent is kept separate from workflow outcome. Judges receive the evidence constructed from the order, delivery, and extracted grievance; missing evidence is not treated automatically as proof for or against a claim.

### Consensus is explicit

For each grievance, three independent judge calls run in parallel. A verdict requires at least two matching judgments. A split without a majority produces an escalation verdict for policy handling.

### Deterministic policy and settlement

LLM judgments do not directly issue refunds. A deterministic policy produces the decision consumed by settlement. Settlement and terminal workflow classification are tracked separately, and execution failure is not a successful outcome.

### Reproducible evaluation

Golden cases live in `data/golden_set.json`; orders and deliveries are stored separately in `data/orders.json` and `data/deliveries.json`. The comparator preserves field-level diagnostics so a case can fail for a specific reason.

## Evaluation

Run the golden set with the configured LLM provider:

```bash
python -m benchmark.runner
```

The full run writes `case_results.json` and `dashboard.json` under `data/benchmark_results/`. It may call the configured provider and incur API charges. To run one case, use `--case CASE-GOLDEN-001`; a focused run does not regenerate the full-run dashboard artifacts.

To rebuild dashboard artifacts offline from a previously saved report:

```bash
python -m benchmark.generate_dashboard --report output.txt
```

This command does not run the agent or call an LLM. The report must be in the format expected by the saved-report loader.

### Reading the metrics

- Simulated CSAT is produced by the benchmark rubric; it is not customer feedback.
- A single run is not a historical trend.
- Latency percentiles use measured cases and include measurement coverage.
- Estimated LLM cost requires complete token telemetry and matching model, provider, endpoint, and prices.
- Estimated savings use explicit manual-handling and review assumptions. They are not realized savings and exclude unmodeled costs such as errors, churn, and infrastructure.

## Interfaces

- **Operations app:** Streamlit UI backed by FastAPI. Reviewers can submit a complaint, inspect the decision, and accept a proposed refund.
- **Evaluation dashboard:** `frontend/refundbench.html` includes saved-run metrics, Dataset Analysis, and AI Agent Workflow diagrams. It reads benchmark artifacts and fixture JSON independently. Serve it over HTTP; direct `file://` access may block JSON loading.

## Data and artifacts

| Path | Purpose |
| --- | --- |
| `data/golden_set.json` | Expected benchmark scenarios and behavior. |
| `data/orders.json` | Order and item fixtures. |
| `data/deliveries.json` | Delivery status and item-level evidence fixtures. |
| `data/refunds.json` | Local refund ledger records. |
| `data/benchmark_results/case_results.json` | Generated case-level results for a saved run. |
| `data/benchmark_results/dashboard.json` | Generated aggregate metrics and dashboard recommendations. |

Benchmark result artifacts are generated output and are git-ignored. The HTML dashboard treats fixture JSON as read-only.

## Tech stack

- Python
- Streamlit
- FastAPI and Uvicorn
- Pydantic
- OpenAI-compatible LLM provider integration
- Pytest
- HTML, CSS, and JavaScript for the standalone dashboard

## Run locally

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure the provider. OpenAI is the default:

```dotenv
OPENAI_API_KEY=your_api_key
OPEN_AI_MODEL=your_model_name
```

For Groq, set `LLM_PROVIDER=GROQ`; `LLM_BASE_URL` is optional. `BACKEND_URL` can override the Streamlit app's API URL (default `http://localhost:8000`).

Start the API and operations app in separate terminals:

```bash
uvicorn backend:app --host 127.0.0.1 --port 8000
```

```bash
streamlit run app.py
```

To serve the standalone HTML dashboard from the repository root:

```bash
python -m http.server 8001
```

Then open `http://127.0.0.1:8001/frontend/refundbench.html`.

## Validation and tests

Validate the fixture dataset without running the agent:

```bash
pytest tests/test_refundbench_dataset.py
```

Run the full test suite:

```bash
pytest
```

## Project structure

```text
app/                    Agent, domain, policy, settlement, and API layers
benchmark/              Golden-case runner, comparator, metrics, artifact tools
config/                 Environment-backed settings
data/                   Golden cases, fixtures, refund ledger, saved artifacts
frontend/app.py         Streamlit operations app and dashboard
frontend/refundbench.html Standalone dashboard, dataset analysis, workflow diagrams
tests/                  Unit, integration, and dataset validation tests
app.py                  Streamlit entry point
backend.py              FastAPI ASGI entry point
```

## Limitations

- This is a demo and benchmark system, not a production refund processor.
- The refund provider is a mock; external payment execution is not integrated.
- Case state is held in memory while the service runs; the local refund ledger is stored in `data/refunds.json`.
- Simulated CSAT and estimated business impact are evaluation proxies, not observed customer or financial results.
