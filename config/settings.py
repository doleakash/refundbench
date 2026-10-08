import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
	openai_api_key: str | None = field(default=None, repr=False)
	open_ai_model: str | None = None
	llm_provider: str = "OPENAI"
	llm_base_url: str | None = None
	backend_url: str = "http://localhost:8000"
	llm_timeout: float = 30.0
	max_llm_retries: int = 1
	max_agent_iterations: int = 10
	max_tool_retries: int = 1
	max_parallel_judge_calls: int = 9
	max_customer_message_length: int = 500
	max_grievances: int = 5
	api_host: str = "127.0.0.1"
	api_port: int = 8000
	log_level: str = "INFO"
	data_dir: Path = PROJECT_ROOT / "data"

	def __post_init__(self) -> None:
		provider = self.llm_provider.strip().upper()
		if provider not in {"OPENAI", "GROQ"}:
			raise ValueError("LLM_PROVIDER must be OPENAI or GROQ")
		object.__setattr__(self, "llm_provider", provider)
		if self.llm_timeout <= 0:
			raise ValueError("LLM_TIMEOUT must be greater than zero")
		if self.max_llm_retries < 0 or self.max_tool_retries < 0:
			raise ValueError("Retry limits cannot be negative")
		if (
			self.max_agent_iterations <= 0
			or self.max_parallel_judge_calls <= 0
			or self.max_customer_message_length <= 0
			or self.max_grievances <= 0
		):
			raise ValueError(
				"Agent limits and customer-message length must be positive"
			)
		if not 1 <= self.api_port <= 65535:
			raise ValueError("API_PORT must be between 1 and 65535")


def get_settings() -> Settings:
	return Settings(
		openai_api_key=os.getenv("OPENAI_API_KEY"),
		open_ai_model=os.getenv("OPEN_AI_MODEL"),
		llm_provider=os.getenv("LLM_PROVIDER", "OPENAI"),
		llm_base_url=os.getenv("LLM_BASE_URL") or None,
		backend_url=os.getenv(
			"BACKEND_URL",
			"http://localhost:8000",
		).rstrip("/"),
		llm_timeout=float(os.getenv("LLM_TIMEOUT", "30")),
		max_llm_retries=int(os.getenv("MAX_LLM_RETRIES", "1")),
		max_agent_iterations=int(os.getenv("MAX_ITERATIONS", "10")),
		max_tool_retries=int(os.getenv("MAX_TOOL_RETRIES", "1")),
		max_parallel_judge_calls=int(
			os.getenv("MAX_PARALLEL_JUDGE_CALLS", "9")
		),
		max_customer_message_length=int(
			os.getenv("MAX_CUSTOMER_MESSAGE_LENGTH", "500")
		),
		max_grievances=int(os.getenv("MAX_GRIEVANCES", "5")),
		api_host=os.getenv("API_HOST", "127.0.0.1"),
		api_port=int(os.getenv("API_PORT", "8000")),
		log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
		data_dir=Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data")),
	)
