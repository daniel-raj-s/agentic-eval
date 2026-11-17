"""AI Agent Package: benchmark e-commerce support agent with RAG and tool calling."""
from .core_agent import AgentResponse, SupportAgent, ToolCallRecord
from .settings import AgentSettings, JudgeSettings

__version__ = "2.0.0"

__all__ = [
    "AgentResponse",
    "AgentSettings",
    "JudgeSettings",
    "SupportAgent",
    "ToolCallRecord",
]
