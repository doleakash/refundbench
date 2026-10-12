import csv
import json
from io import StringIO
from pathlib import Path

import requests
import streamlit as st

from frontend.api_client import RefundBenchAPIClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_ARTIFACT_PATH = (
    PROJECT_ROOT / "data" / "benchmark_results" / "dashboard.json"
)
CASE_ARTIFACT_PATH = (
    PROJECT_ROOT / "data" / "benchmark_results" / "case_results.json"
)
EXAMPLE_COMPLAINT = (
    "My order was delivered 85 minutes late.\n"
    "Two of my four biryanis were missing.\n"
    "The raita container leaked everywhere."
)
CUSTOM_ORDER_OPTION = "Enter a custom order ID"


def truncate_display_text(value: str, max_length: int) -> str:
    if len(value) <= max_length:
       return value
    return value[: max_length - 1].rstrip() + "…"


def dashboard_cases_csv(cases: list[dict]) -> str:
   output = StringIO(newline="")
   fields = (
      ("Case ID", "case_id"),
      ("Customer Message", "customer_issue"),
      ("Agent Decision", "display_actual_decision"),
      ("Expected Outcome", "display_expected_outcome"),
      ("Resolution Time", "display_latency"),
      ("Simulated CSAT", "display_simulated_csat"),
   )
   writer = csv.DictWriter(
      output,
      fieldnames=[label for label, _ in fields],
      extrasaction="ignore",
   )
   writer.writeheader()
   for case in cases:
      writer.writerow({
         label: case.get(key, "")
         for label, key in fields
      })
   return output.getvalue()


def load_dashboard_artifacts(
    dashboard_path: Path = DASHBOARD_ARTIFACT_PATH,
    case_path: Path = CASE_ARTIFACT_PATH,
) -> tuple[dict, list[dict]]:
    with dashboard_path.open(encoding="utf-8") as dashboard_file:
       dashboard = json.load(dashboard_file)
    with case_path.open(encoding="utf-8") as case_file:
       cases = json.load(case_file)
    if dashboard.get("schema_version") != 1 or not isinstance(cases, list):
       raise ValueError("Unsupported or invalid dashboard artifact schema.")
    if dashboard.get("case_count") != len(cases):
       raise ValueError("Dashboard and case artifacts do not match.")
    return dashboard, cases


def dashboard_artifact_empty_state() -> str:
    return (
       "No dashboard artifact was found. If `output.txt` contains a saved "
       "evaluation, generate it offline with `python -m "
       "benchmark.generate_dashboard`, then launch the dashboard with "
       "`streamlit run app.py`. If no saved evaluation exists, run the "
       "benchmark separately first; that evaluation may call the configured "
       "LLM provider. Opening or refreshing this dashboard never runs it."
    )


def render_dashboard_bar_chart(
    data: list[dict],
    *,
    x_field: str,
    y_field: str,
    height: int,
    horizontal: bool = False,
    color_field: str | None = None,
    color_domain: list[str] | None = None,
    color_range: list[str] | None = None,
    legend_title: str | None = None,
) -> None:
    """Render the dashboard's small bar charts with a consistent light theme."""
    axis_style = {
       "labelColor": "#53627a",
       "titleColor": "#53627a",
       "labelFontSize": 11,
       "titleFontSize": 11,
       "gridColor": "#edf1f5",
       "domainColor": "#dce3eb",
       "tickColor": "#dce3eb",
    }
    quantitative = {
       "type": "quantitative",
       "axis": {**axis_style, "grid": True, "format": "~s"},
    }
    nominal = {
       "type": "nominal",
       "axis": {
          **axis_style,
          "grid": False,
          "labelLimit": 180,
          "labelOverlap": False,
       },
    }
    encoding = {
       "x": (
          {"field": y_field, **quantitative}
          if horizontal
          else {"field": x_field, **nominal}
       ),
       "y": (
          {"field": x_field, **nominal}
          if horizontal
          else {"field": y_field, **quantitative}
       ),
       "tooltip": (
          [
             {"field": y_field, "type": "quantitative"},
             {"field": x_field, "type": "nominal"},
          ]
          if horizontal
          else [
             {"field": x_field, "type": "nominal"},
             {"field": y_field, "type": "quantitative"},
          ]
       ),
    }
    if color_field is not None:
       encoding["color"] = {
          "field": color_field,
          "type": "nominal",
          "scale": {
             "domain": color_domain or [
                "Automated", "Escalated", "Total cases",
             ],
             "range": color_range or [
                "#22a06b", "#ed8a35", "#a7b1c2",
             ],
          },
          "legend": {
             "title": legend_title,
             "orient": "top",
             "direction": "horizontal",
             "anchor": "end",
             "labelColor": "#53627a",
             "labelFontSize": 10,
             "symbolType": "circle",
             "symbolStrokeColor": "#dce3eb",
          },
       }
    chart_config = {
       "background": "#ffffff",
       "view": {"stroke": None, "fill": "#ffffff"},
       "font": "Arial",
       "axis": {
          **axis_style,
          "grid": True,
          "gridColor": "#edf1f5",
          "gridWidth": 1,
       },
       "legend": {
          "labelColor": "#53627a",
          "titleColor": "#53627a",
          "labelFont": "Arial",
          "titleFont": "Arial",
          "labelFontSize": 10,
          "titleFontSize": 10,
          "symbolStrokeColor": "#dce3eb",
       },
       "bar": {"cornerRadiusEnd": 3},
    }
    spec = {
       "mark": {"type": "bar", "tooltip": True},
       "encoding": encoding,
       "height": height,
       "config": chart_config,
    }
    st.vega_lite_chart(
       data,
       spec,
       use_container_width=True,
       theme=None,
    )


def render_dashboard() -> None:
    st.markdown(
       """
       <style>
       :root {
          color-scheme: light;
       }
       html, body, [data-testid="stApp"],
       [data-testid="stAppViewContainer"],
       [data-testid="stMain"], section.main {
          background: #f4f7fb !important;
          color: #172033 !important;
       }
       .block-container {
          width: 100%;
          max-width: 1600px;
          padding: 0.35rem 1.75rem 1.1rem !important;
       }
       [data-testid="stHeader"] {
          background: transparent !important;
       }
       [data-testid="stMainBlockContainer"] {
          padding-top: 0.35rem !important;
       }
       [data-testid="stVerticalBlock"] {
          gap: 0.65rem;
       }
       [data-testid="stHorizontalBlock"] {
          gap: 0.85rem;
       }
       [data-testid="stElementContainer"] {
          margin-bottom: 0;
       }
       h1 {
          color: #172033;
          font-size: 1.8rem !important;
          font-weight: 700 !important;
          letter-spacing: -0.025em;
          margin-bottom: 0.15rem !important;
       }
       h2, h3 {
          color: #172033;
          font-weight: 650 !important;
          letter-spacing: -0.01em;
       }
       h3 {
          font-size: 1.02rem !important;
          margin: 0.38rem 0 0.18rem !important;
       }
       [data-testid="stMetric"] {
          background: transparent;
          border: 0;
          padding: 0;
          box-shadow: none;
       }
       [data-testid="stMetricLabel"] {
          color: #61708a;
          font-size: 0.79rem;
          font-weight: 500;
       }
       [data-testid="stMetricValue"] {
          color: #172033;
          font-size: 1.48rem;
          font-weight: 650;
       }
       [data-testid="stMetricDelta"] {
          font-size: 0.75rem;
       }
       .dashboard-section-title {
          color: #23314a;
          font-size: 0.94rem;
          font-weight: 650;
          margin: 0 0 0.12rem;
       }
       .dashboard-section-note {
          color: #718096;
          font-size: 0.78rem;
          margin: 0 0 0.12rem;
       }
       [data-testid="stVerticalBlockBorderWrapper"] {
          background: #fff;
          border-color: #e5eaf1;
          border-radius: 9px;
          box-shadow: 0 1px 3px rgba(24, 39, 75, 0.035);
       }
       [data-testid="stVerticalBlockBorderWrapper"] > div {
          padding: 0.68rem 0.78rem 0.52rem;
       }
       .st-key-dashboard-kpi-savings,
       .st-key-dashboard-kpi-csat,
       .st-key-dashboard-kpi-errors,
       .st-key-dashboard-kpi-time {
          background: #fff;
          border: 1px solid #e2e8f0;
          border-radius: 12px;
          padding: 1rem 1rem 0.8rem;
          box-shadow: 0 1px 3px rgba(24, 39, 75, 0.04);
          min-height: 218px;
       }
       .st-key-dashboard-kpi-savings [data-testid="stMetricValue"],
       .st-key-dashboard-kpi-csat [data-testid="stMetricValue"],
       .st-key-dashboard-kpi-errors [data-testid="stMetricValue"],
       .st-key-dashboard-kpi-time [data-testid="stMetricValue"] {
          color: #101a2e;
          font-size: 2rem;
          font-weight: 700;
          line-height: 1.15;
       }
       .st-key-dashboard-kpi-savings [data-testid="stMetric"],
       .st-key-dashboard-kpi-csat [data-testid="stMetric"],
       .st-key-dashboard-kpi-errors [data-testid="stMetric"],
       .st-key-dashboard-kpi-time [data-testid="stMetric"] {
          padding: 0.35rem 0 0.4rem;
       }
       .st-key-dashboard-kpi-savings [data-testid="stMetricLabel"],
       .st-key-dashboard-kpi-csat [data-testid="stMetricLabel"],
       .st-key-dashboard-kpi-errors [data-testid="stMetricLabel"],
       .st-key-dashboard-kpi-time [data-testid="stMetricLabel"] {
          color: #63718a;
          font-size: 0.78rem;
       }
       .dashboard-kpi-heading {
          display: flex;
          align-items: center;
          gap: 0.6rem;
          min-height: 42px;
          color: #1b2941;
          font-size: 0.9rem;
          font-weight: 650;
          line-height: 1.2;
       }
       .dashboard-kpi-icon {
          width: 40px;
          height: 40px;
          flex: 0 0 40px;
          display: grid;
          place-items: center;
          border-radius: 10px;
       }
       .dashboard-kpi-icon svg {
          width: 21px;
          height: 21px;
       }
       .dashboard-kpi-info {
          margin-left: auto;
          color: #91a0b5;
          font-size: 0.85rem;
          font-weight: 500;
       }
       .dashboard-kpi-divider {
          height: 1px;
          margin: 0.35rem 0 0.55rem;
          background: #edf0f4;
       }
       .dashboard-kpi-support {
          display: flex;
          justify-content: space-between;
          gap: 0.4rem;
          color: #6d7b90;
          font-size: 0.69rem;
          line-height: 1.35;
       }
       .dashboard-kpi-support strong {
          display: block;
          margin-top: 0.15rem;
          color: #27364e;
          font-size: 0.78rem;
          font-weight: 600;
       }
       .st-key-dashboard-kpi-errors .dashboard-kpi-support,
       .st-key-dashboard-kpi-time .dashboard-kpi-support {
          font-size: 0.68rem;
       }
       .dashboard-kpi-stack {
          display: flex;
          width: 100%;
          height: 10px;
          overflow: hidden;
          margin: 0.1rem 0 0.5rem;
          border-radius: 999px;
          background: #edf1f5;
       }
       .dashboard-kpi-stack-segment {
          height: 100%;
       }
       .dashboard-kpi-ratings {
          display: flex;
          justify-content: space-between;
          gap: 0.15rem;
          color: #62718a;
          font-size: 0.67rem;
       }
       .dashboard-kpi-ratings strong {
          display: block;
          color: #26354d;
          font-size: 0.75rem;
       }
       .st-key-dashboard-kpi-errors [data-testid="stMetricValue"],
       .st-key-dashboard-kpi-time [data-testid="stMetricValue"] {
          font-size: 1.8rem;
       }
       .dashboard-positive { color: #159766; }
       .dashboard-risk { color: #d94b4b; }
       .dashboard-neutral { color: #3975d5; }
       .dashboard-warning { color: #c98222; }
       [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"] {
          color: #718096;
          font-size: 0.78rem;
          line-height: 1.4;
       }
       [data-testid="stHorizontalBlock"] {
          gap: 0.72rem;
       }
       div[data-testid="stDataFrame"] {
          border: 1px solid #edf0f4;
          border-radius: 8px;
          background: #fff;
       }
       div[data-testid="stDataFrame"] [role="columnheader"] {
          background: #f7f9fc;
          color: #43516a;
          font-weight: 600;
          border-bottom: 1px solid #e5eaf1;
       }
       div[data-testid="stDataFrame"] [role="gridcell"] {
          color: #344258;
          font-size: 0.8rem;
       }
       [data-testid="stVegaLiteChart"],
       [data-testid="stVegaLiteChart"] > div,
       [data-testid="stVegaLiteChart"] svg,
       [data-testid="stVegaLiteChart"] canvas {
          background: #fff !important;
          color-scheme: light !important;
       }
       [data-testid="stVegaLiteChart"] {
          padding: 0 !important;
          margin: 0 !important;
          border-radius: 8px;
          overflow: hidden;
       }
       [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
          gap: 0.38rem;
       }
       [data-testid="stHorizontalBlock"] {
          align-items: stretch;
       }
       [data-testid="stVerticalBlockBorderWrapper"] {
          background: #fff !important;
       }
       </style>
       """,
       unsafe_allow_html=True,
    )
    st.title("Refund Operations Dashboard")
    st.caption(
       "Performance, accuracy, cost and customer impact of the AI refund agent."
    )
    try:
       dashboard, cases = load_dashboard_artifacts()
    except FileNotFoundError:
       st.info(dashboard_artifact_empty_state())
       return
    except (OSError, json.JSONDecodeError, ValueError) as error:
       st.info(f"Not available: dashboard artifacts could not be loaded ({error}).")
       return

    metrics = dashboard["metrics"]
    assumptions = dashboard["assumptions"]
    currency = assumptions["cost_currency"]
    decision_metrics = metrics.get("decision_metrics", {})
    st.caption(
       f"{dashboard['case_count']} evaluated cases · "
       f"{dashboard['trend_note']}"
    )
    kpis = st.columns(4)
    net_savings = metrics["estimated_net_savings"]
    csat = metrics["simulated_csat"]
    incorrect = metrics["incorrect_refund_decisions"]
    p50 = metrics["p50_latency_ms"]
    p95 = metrics["p95_latency_ms"]
    confusion = metrics["decision_confusion_matrix"]

    def kpi_heading(title: str, icon: str, tile_color: str) -> str:
       return (
          '<div class="dashboard-kpi-heading">'
          f'<span class="dashboard-kpi-icon" style="background:{tile_color}">'
          f"{icon}</span><span>{title}</span>"
          '<span class="dashboard-kpi-info" title="Based on the saved benchmark run">ⓘ</span>'
          "</div>"
       )

    def kpi_support(items: list[tuple[str, str]]) -> str:
       return (
          '<div class="dashboard-kpi-divider"></div>'
          '<div class="dashboard-kpi-support">'
          + "".join(
             f"<span>{label}<strong>{value}</strong></span>"
             for label, value in items
          )
          + "</div>"
       )

    with kpis[0]:
       with st.container(key="dashboard-kpi-savings", border=True):
          st.markdown(
             kpi_heading(
                "Estimated Net Savings",
                '<svg viewBox="0 0 24 24" fill="none" stroke="#159766" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>'
                "</svg>",
                "#dcf7e9",
             ),
             unsafe_allow_html=True,
          )
          st.metric(
             "Estimated net savings",
             "Not available"
             if net_savings is None
             else f"{currency} {net_savings:,.2f}",
             label_visibility="hidden",
          )
          savings_support = [
             (
                "Manual handling assumption",
                (
                   f"{currency} {assumptions['manual_handling_cost_per_case']:.2f}/case"
                   if assumptions.get("manual_handling_cost_per_case") is not None
                   else "Not configured"
                ),
             ),
             (
                "Handling cost avoided",
                (
                   f"{currency} {metrics['estimated_manual_handling_cost_avoided']:.4f}"
                   if metrics.get("estimated_manual_handling_cost_avoided") is not None
                   else "Not available"
                ),
             ),
             (
                "Human-review cost",
                (
                   f"{currency} {metrics['estimated_human_review_cost']:.4f}"
                   if metrics.get("estimated_human_review_cost") is not None
                   else "Not available"
                ),
             ),
          ]
          if metrics["estimated_llm_cost"] is not None:
             savings_support.append(
                ("AI cost", f"{currency} {metrics['estimated_llm_cost']:.4f}")
             )
          st.markdown(
             kpi_support(savings_support),
             unsafe_allow_html=True,
          )
          st.caption(
             assumptions.get(
                "business_impact_methodology",
                "Estimated savings are assumption-based, not realized savings.",
             )
          )
          if metrics.get("savings_unavailable_reasons"):
             st.caption("Savings unavailable: " + " ".join(
                metrics["savings_unavailable_reasons"]
             ))
    with kpis[1]:
       with st.container(key="dashboard-kpi-csat", border=True):
          st.markdown(
             kpi_heading(
                "Simulated Customer Satisfaction",
                '<svg viewBox="0 0 24 24" fill="none" stroke="#3975d5" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>'
                '<circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/>'
                '<path d="M16 3.13a4 4 0 0 1 0 7.75"/>'
                "</svg>",
                "#e6edff",
             ),
             unsafe_allow_html=True,
          )
          st.metric(
             "Simulated CSAT",
             "Not available" if csat is None else f"{csat:.1f} / 5",
             label_visibility="hidden",
          )
          distribution = metrics["csat_distribution"]
          if distribution:
             rating_colors = ["#20b876", "#8edeb5", "#cbd3df", "#f36b6b"]
             total_ratings = sum(item["cases"] for item in distribution)
             stack_segments = "".join(
                '<span class="dashboard-kpi-stack-segment" '
                f'style="width:{item["cases"] / total_ratings * 100:.2f}%;'
                f'background:{rating_colors[min(index, len(rating_colors) - 1)]}"></span>'
                for index, item in enumerate(distribution)
             )
             rating_labels = "".join(
                "<span>"
                f'{item["rating"]} stars'
                f'<strong>{item["cases"] / total_ratings:.0%}</strong>'
                "</span>"
                for item in distribution
             )
             st.markdown(
                '<div class="dashboard-kpi-divider"></div>'
                f'<div class="dashboard-kpi-stack">{stack_segments}</div>'
                f'<div class="dashboard-kpi-ratings">{rating_labels}</div>',
                unsafe_allow_html=True,
             )
          else:
             st.markdown(
                kpi_support([("Feedback", "Distribution not available")]),
                unsafe_allow_html=True,
             )
          st.caption("Simulated rubric · not actual customer feedback")
    with kpis[2]:
       with st.container(key="dashboard-kpi-errors", border=True):
          st.markdown(
             kpi_heading(
                "Incorrect Refund Decisions",
                '<svg viewBox="0 0 24 24" fill="none" stroke="#e24d57" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11z"/>'
                '<path d="M12 8v4m0 4h.01"/>'
                "</svg>",
                "#ffeaeb",
             ),
             unsafe_allow_html=True,
          )
          incorrect_available = (
             incorrect["false_positives"] is not None
             and incorrect["false_negatives"] is not None
             and all(value is not None for value in confusion.values())
          )
          classified_cases = (
             sum(confusion.values()) if incorrect_available else 0
          )
          error_rate = (
             (
                incorrect["false_positives"]
                + incorrect["false_negatives"]
             )
             / classified_cases
             if classified_cases
             else None
          )
          st.metric(
             "Incorrect decision rate",
             f"{error_rate:.1%}" if error_rate is not None else "Not available",
             label_visibility="hidden",
          )
          fp_value = (
             str(incorrect["false_positives"])
             if incorrect["false_positives"] is not None
             else "N/A"
          )
          fn_value = (
             str(incorrect["false_negatives"])
             if incorrect["false_negatives"] is not None
             else "N/A"
          )
          st.markdown(
             kpi_support(
                [
                   ("False positives", fp_value),
                   ("False negatives", fn_value),
                ]
             ),
             unsafe_allow_html=True,
          )
          st.caption(
             f"{decision_metrics.get('decision_confusion_matrix_denominator', 0)} "
             "cases in the binary matrix; "
             f"{decision_metrics.get('decision_confusion_matrix_excluded_expected_ambiguous_count', 0)} "
             "excluded for ambiguous expected labels and "
             f"{decision_metrics.get('decision_confusion_matrix_excluded_actual_ambiguous_count', 0)} "
             "for ambiguous actual labels. Agent escalations count as not "
             "automatically approved."
          )
    with kpis[3]:
       with st.container(key="dashboard-kpi-time", border=True):
          st.markdown(
             kpi_heading(
                "Resolution Time",
                '<svg viewBox="0 0 24 24" fill="none" stroke="#20a9d2" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>'
                "</svg>",
                "#e2f7fc",
             ),
             unsafe_allow_html=True,
          )
          st.metric(
             "P95 measured agent processing",
             f"{p95 / 1000:.1f} s" if p95 is not None else "Not available",
             label_visibility="hidden",
          )
          st.markdown(
             '<div class="dashboard-kpi-divider"></div>',
             unsafe_allow_html=True,
          )
          st.markdown(
             kpi_support(
                [
                   (
                      "P50 measured",
                      f"{p50 / 1000:.1f} s" if p50 is not None else "N/A",
                   ),
                   (
                      "P95 measured",
                      f"{p95 / 1000:.1f} s" if p95 is not None else "N/A",
                   ),
                   (
                      "Latency coverage",
                         (
                            f"{metrics.get('latency_measured_case_count', 0)}/"
                            f"{dashboard['case_count']} cases"
                         ),
                   ),
                ]
             ),
             unsafe_allow_html=True,
          )

    def panel_title(title: str, note: str | None = None) -> None:
       st.markdown(
          f'<div class="dashboard-section-title">{title}</div>',
          unsafe_allow_html=True,
       )
       if note:
          st.markdown(
             f'<div class="dashboard-section-note">{note}</div>',
             unsafe_allow_html=True,
          )

    def unavailable(note: str) -> None:
       st.caption(f"Not available · {note}")

    def overview_title(icon: str, title: str, note: str) -> None:
       st.markdown(
          '<div style="display:flex;align-items:center;gap:.55rem;'
          'margin-bottom:.25rem">'
          '<span style="display:flex;width:25px;height:25px;'
          'align-items:center;justify-content:center;color:#3975d5">'
          f"{icon}</span>"
          '<span style="color:#1b2941;font-size:.94rem;font-weight:650">'
          f"{title}</span></div>"
          f'<div class="dashboard-section-note">{note}</div>',
          unsafe_allow_html=True,
       )

    overview_left, overview_right = st.columns(2)
    with overview_left:
       with st.container(border=True):
          overview_title(
             '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
             'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
             'stroke-linejoin="round"><path d="M3 3v18h18"/>'
             '<path d="m19 9-5 5-4-4-5 5"/></svg>',
             "Automation vs. Escalation",
             "Single benchmark run · case counts, not a historical trend",
          )
          outcome_data = [
             {"Outcome": "Automated", "Cases": metrics["automated_count"]},
             {"Outcome": "Escalated", "Cases": metrics["escalated_count"]},
             {
                "Outcome": "Processing",
                "Cases": dashboard["case_count"]
                         - metrics["automated_count"]
                         - metrics["escalated_count"],
             },
          ]
          render_dashboard_bar_chart(
             outcome_data,
             x_field="Outcome",
             y_field="Cases",
             color_field="Outcome",
             color_domain=["Automated", "Escalated", "Processing"],
             color_range=["#22a06b", "#ed8a35", "#3975d5"],
             legend_title=None,
             height=185,
             horizontal=True,
          )
    with overview_right:
       with st.container(border=True):
          overview_title(
             '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
             'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
             'stroke-linejoin="round"><path d="m12 3 2.7 5.5 6.1.9-4.4 4.3 '
             '1 6.1-5.4-2.9-5.4 2.9 1-6.1-4.4-4.3 6.1-.9L12 3z"/></svg>',
             "CSAT Rating Distribution",
             (
                "Simulated rubric · not actual feedback; "
                f"{metrics.get('simulated_csat_unscored_case_count', 0)} "
                "case(s) unscored"
             ),
          )
          if metrics["csat_distribution"]:
             rating_domain = [
                item["rating"]
                for item in sorted(
                   metrics["csat_distribution"],
                   key=lambda item: float(item["rating"]),
                   reverse=True,
                )
             ]
             rating_colors = [
                "#20b876"
                if float(rating) >= 4.5
                else "#8edeb5"
                if float(rating) >= 4
                else "#cbd3df"
                if float(rating) >= 3
                else "#f36b6b"
                for rating in rating_domain
             ]
             render_dashboard_bar_chart(
                metrics["csat_distribution"],
                x_field="rating",
                y_field="cases",
                color_field="rating",
                color_domain=rating_domain,
                color_range=rating_colors,
                legend_title="Rating",
                height=185,
             )
          else:
             unavailable("no simulated CSAT scores are available")

    reason_col, matrix_col, time_col = st.columns(3)
    with reason_col:
       with st.container(border=True):
          panel_title("Top escalation reasons", "Recorded grievance types")
          if metrics["top_escalation_reasons"]:
             render_dashboard_bar_chart(
                metrics["top_escalation_reasons"],
                x_field="reason",
                y_field="count",
                horizontal=True,
                height=190,
             )
          else:
             unavailable("no escalation reasons were recorded")
    with matrix_col:
       with st.container(border=True):
          panel_title(
             "Refund Decision Quality",
             "Expected compared with actual",
          )
          if metrics["confusion_matrix_rows"] is not None:
             st.dataframe(
                metrics["confusion_matrix_rows"],
                hide_index=True,
                use_container_width=True,
                height=180,
             )
             st.caption(
                f"Matrix denominator: "
                f"{decision_metrics.get('decision_confusion_matrix_denominator', 0)}. "
                f"{decision_metrics.get('decision_confusion_matrix_excluded_expected_ambiguous_count', 0)} "
                "case(s) excluded for ambiguous expected labels; "
                f"{decision_metrics.get('decision_confusion_matrix_excluded_actual_ambiguous_count', 0)} "
                "for ambiguous actual labels. Agent escalations are classified "
                "as not automatically approved."
             )
          else:
             unavailable("expected or actual decision data is incomplete")
    with time_col:
       with st.container(border=True):
          panel_title(
             "Agent Resolution-Time Distribution",
             "Measured processing time",
          )
          if metrics["latency_distribution"]:
             render_dashboard_bar_chart(
                metrics["latency_distribution"],
                x_field="duration",
                y_field="cases",
                height=190,
             )
          else:
             unavailable("no measured agent timings were recorded")

    with st.container(border=True):
       panel_title(
          "Execution Reliability",
          "Observed calls, retries, and benchmark execution failures",
       )
       reliability_cols = st.columns(4)

       def display_count(value: int | None) -> str:
          return f"{value:,}" if value is not None else "Not available"

       reliability_cols[0].metric(
          "LLM calls", display_count(metrics.get("llm_call_count"))
       )
       reliability_cols[1].metric(
          "Judge calls", display_count(metrics.get("judge_call_count"))
       )
       reliability_cols[2].metric(
          "Failed LLM calls",
          display_count(metrics.get("failed_llm_call_count")),
       )
       reliability_cols[3].metric(
          "Execution failures",
          display_count(metrics.get("execution_failure_count")),
       )
       retry_text = (
          f"{display_count(metrics.get('llm_retry_count'))} LLM · "
          f"{display_count(metrics.get('tool_retry_count'))} tool retries"
       )
       coverage = metrics.get("llm_telemetry_coverage", {})
       st.caption(
          f"Retries: {retry_text} · LLM telemetry coverage: "
          f"{display_count(coverage.get('instrumented_case_count'))}/"
          f"{dashboard['case_count']} cases instrumented; "
          f"{display_count(coverage.get('uninstrumented_case_count'))} "
          "missing"
       )

    recent_col, action_col = st.columns([1.65, 1], gap="small")
    with recent_col:
       with st.container(border=True):
          panel_title("Recent Cases", "Latest cases in this saved run")
          st.dataframe(
             [
                {
                   "Case ID": case["case_id"],
                   "Customer issue": (
                      truncate_display_text(case["customer_issue"], 72)
                   ),
                   "Agent decision": case["display_actual_decision"],
                   "Expected outcome": truncate_display_text(
                      case["display_expected_outcome"], 58
                   ),
                   "Resolution time": case["display_latency"],
                   "Simulated CSAT": case["display_simulated_csat"],
                }
                for case in cases[-10:]
             ],
             hide_index=True,
             use_container_width=True,
             height=310,
             column_config={
                "Case ID": st.column_config.TextColumn(
                   "Case ID", width="small"
                ),
                "Customer issue": st.column_config.TextColumn(
                   "Customer issue", width="large",
                   help="Long messages are shortened for readability.",
                ),
                "Agent decision": st.column_config.TextColumn(
                   "Agent decision", width="medium"
                ),
                "Expected outcome": st.column_config.TextColumn(
                   "Expected outcome", width="medium",
                   help="Long summaries are shortened for readability.",
                ),
                "Resolution time": st.column_config.TextColumn(
                   "Resolution time", width="small"
                ),
                "Simulated CSAT": st.column_config.TextColumn(
                   "Simulated CSAT", width="small"
                ),
             },
          )
    with action_col:
       with st.container(border=True):
          panel_title("Recommended Actions", "Prioritized for follow-up")
          for action in dashboard["recommendations"]:
             st.markdown(
                f"**{action['priority']} · {action['title']}**"
             )
             st.caption(action["explanation"])
             if action["case_ids"]:
                st.caption("Cases: " + ", ".join(action["case_ids"]))
             st.divider()


def api_error_message(error: requests.RequestException) -> str:
    if isinstance(error, (requests.ConnectionError, requests.Timeout)):
       return "Unable to connect to RefundBench backend."
    response = error.response
    if response is not None:
       try:
          detail = response.json().get("detail")
       except ValueError:
          detail = None
       if isinstance(detail, str):
          return detail
    return "RefundBench could not complete the request. Please try again."


def render_case(case: dict, api_client: RefundBenchAPIClient) -> None:
    st.subheader("Case result")
    case_col, order_col, status_col = st.columns(3)
    case_col.metric("Case ID", case["case_id"])
    order_col.metric("Order ID", case["order_id"])
    status = case["status"]
    status_col.metric("Outcome", status.replace("_", " "))

    settlement = case.get("settlement")
    approved = (
       settlement is not None
       and settlement["status"] == "AUTO_APPROVED"
       and settlement["amount"] > 0
    )
    if approved:
       st.success("REFUND APPROVED")
       st.metric("Refund amount", f"₹{settlement['amount']:,.0f}")
    elif status == "ESCALATED" or (
       settlement is not None and settlement["status"] == "ESCALATE"
    ):
       st.warning("CASE ESCALATED")
       if case.get("escalation_reason"):
          st.write(case["escalation_reason"])
       elif settlement and settlement.get("reason"):
          st.write(settlement["reason"])
    if case.get("response"):
       st.markdown("**Customer response**")
       st.info(case["response"])
    if approved:
       accepted_refund = st.session_state.get("accepted_refund")
       if accepted_refund and accepted_refund.get("case_id") == case["case_id"]:
          st.success(f"Refund status: {accepted_refund['status']}")
          st.caption(f"Idempotency key: {accepted_refund['idempotency_key']}")
       elif st.button("Accept Refund", type="primary"):
          try:
             with st.spinner("Processing refund acceptance..."):
                st.session_state["accepted_refund"] = (
                   api_client.accept_refund(case["case_id"])
                )
             st.rerun()
          except requests.RequestException as error:
             st.error(api_error_message(error))

    grievances = case.get("grievances", [])
    if grievances:
       st.subheader("Grievances")
       for grievance in grievances:
          label = grievance["type"].replace("_", " ").title()
          with st.container(border=True):
             st.markdown(f"**{grievance['grievance_id']} · {label}**")
             st.write(grievance["claim"])
             evidence = grievance.get("evidence")
             decision = grievance.get("policy_decision")
             left, right = st.columns(2)
             if evidence:
                with left:
                   st.markdown(f"**Evidence · {evidence['source']}**")
                   for fact in evidence["facts"]:
                      st.write(f"- {fact}")
             with right:
                if grievance.get("consensus"):
                   st.markdown(f"**Consensus:** {grievance['consensus']}")
                if decision:
                   st.markdown(f"**Policy:** {decision['action']}")
                   if decision["action"] == "REFUND":
                      st.write(f"₹{decision['refund_amount']:,.0f} approved")
                   if decision.get("reason"):
                      st.caption(decision["reason"])
    with st.expander("Agent Execution"):
       for action in case.get("actions", []):
          st.write(f"✓ {action}")
       with st.expander("Observations"):
          for observation in case.get("observations", []):
             st.write(f"- {observation}")
    if case.get("tool_errors"):
       with st.expander("Tool / system errors"):
          for error in case["tool_errors"]:
             st.error(error)


def main() -> None:
    st.set_page_config(page_title="RefundBench", page_icon="🧾", layout="wide")
    view = st.sidebar.radio(
       "Workspace",
       ["Refund Operations Dashboard", "Process a customer case"],
    )
    if view == "Refund Operations Dashboard":
       render_dashboard()
       return
    api_client = RefundBenchAPIClient()
    st.title("RefundBench")
    st.caption("AI-powered customer support & refund resolution")
    st.header("Customer request")
    try:
       order_ids = api_client.get_orders()
    except requests.RequestException:
       order_ids = []
       st.info("Order list unavailable; enter an order ID manually.")
    if order_ids:
       order_choice = st.selectbox(
          "Order ID",
          [*order_ids, CUSTOM_ORDER_OPTION],
       )
       order_id = (
          st.text_input("Custom order ID")
          if order_choice == CUSTOM_ORDER_OPTION
          else order_choice
       )
    else:
       order_id = st.text_input("Order ID")
    complaint = st.text_area(
       "Customer complaint",
       value=EXAMPLE_COMPLAINT,
       height=150,
    )
    if st.button("Process Complaint", type="primary"):
       if not order_id.strip():
          st.error("Enter an order ID before processing the complaint.")
       elif not complaint.strip():
          st.error("Enter a customer complaint before processing.")
       else:
          st.session_state.pop("case_result", None)
          st.session_state.pop("accepted_refund", None)
          try:
             with st.spinner("Agent is analyzing the case..."):
                created_case = api_client.create_case(
                   order_id=order_id.strip(),
                   customer_message=complaint.strip(),
                )
                st.session_state["case_result"] = api_client.get_case(
                   created_case["case_id"]
                )
          except requests.RequestException as error:
             st.error(api_error_message(error))
    if case := st.session_state.get("case_result"):
       render_case(case, api_client)


if __name__ == "__main__":
    main()
