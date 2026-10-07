from __future__ import annotations

from typing import Protocol

from app.agent.state import AgentState


class OrderRepository(Protocol):
	def list_order_ids(self) -> list[str]: ...


class CaseRepository(Protocol):
	def save(self, state: AgentState) -> None: ...

	def get(self, case_id: str) -> AgentState | None: ...


class InMemoryCaseRepository:
	def __init__(self):
		self._cases: dict[str, AgentState] = {}

	def save(self, state: AgentState) -> None:
		self._cases[state.case_id] = state

	def get(self, case_id: str) -> AgentState | None:
		return self._cases.get(case_id)
