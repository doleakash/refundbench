from app.agent.orchestrator import run_agent


def main() -> None:
	message = """
My order was delivered 85 minutes late.
Two of my four biryanis were missing.
The raita container leaked everywhere.
"""
	state = run_agent(
		customer_message=message,
		order_id="ORD-123",
	)
	print(state.response)


if __name__ == "__main__":
	main()
