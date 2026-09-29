"""Public API for the TrustedAI movie agent."""

from .agent import MovieAgent
from .config import AgentConfig
from .schemas import AgentRequest, AgentResponse, Recommendation

__all__ = ["AgentConfig", "AgentRequest", "AgentResponse", "MovieAgent", "Recommendation"]
