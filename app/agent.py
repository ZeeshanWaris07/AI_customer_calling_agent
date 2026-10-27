
from typing import Literal, Optional

from pydantic import BaseModel, Field


Intent = Literal[
    "greeting",
    "faq",
    "order_status",
    "complaint",
    "human_handoff",
    "unknown",
]

Outcome = Literal[
    "resolved",
    "escalated",
    "needs_clarification",
]


class AgentRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="The customer's message",
    )
    session_id: Optional[str] = None


class AgentResponse(BaseModel):
    session_id: Optional[str] = None
    intent: Intent
    response: str = Field(min_length=1)
    outcome: Outcome
