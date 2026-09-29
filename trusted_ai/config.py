from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "openai:deepseek/deepseek-v4.1-flash"  # langchain id, served through OPENAI_BASE_URL
    max_results: int = 2            # most movies a recommendation tool may return per call
    max_agent_steps: int = 10       # langgraph recursion limit per turn
    history_token_limit: int = 100_000  # context size (last call's input tokens) that triggers a summary
    keep_recent_turns: int = 2      # turns kept verbatim after the first one when summarising
