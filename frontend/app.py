import requests
import streamlit as st

from frontend.api_client import RefundBenchAPIClient

EXAMPLE_COMPLAINT = (
    "My order was delivered 85 minutes late.\n"
    "Two of my four biryanis were missing.\n"
    "The raita container leaked everywhere."
)
CUSTOM_ORDER_OPTION = "Enter a custom order ID"


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
            st.caption(
                f"Idempotency key: {accepted_refund['idempotency_key']}"
            )
        else:
            st.markdown("**Would you like to proceed with the refund?**")
            if st.button("Accept Refund", type="primary"):
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
    api_client = RefundBenchAPIClient()
    st.set_page_config(page_title="RefundBench", page_icon="🧾", layout="wide")
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
