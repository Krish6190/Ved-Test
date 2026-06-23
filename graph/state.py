from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, Field

def limit_messages(left: Sequence[BaseMessage], right: Sequence[BaseMessage]) -> list[BaseMessage]:
    """
    Combines messages. Keeps the permanent system prompt at index 0,
    and limits the rest to the 38 most recent messages to protect VRAM.
    """
    combined = add_messages(left, right)
    if len(combined) <= 40:
        return combined
    system_prompt = None
    pinned_messages = []
    standard_messages = []
    for msg in combined:
        if isinstance(msg, SystemMessage) and system_prompt is None:
            system_prompt = msg
        elif msg.additional_kwargs.get("pinned") is True:
            pinned_messages.append(msg)
        else:
            standard_messages.append(msg)
    max_standard = max(2, 38 - len(pinned_messages))
    recent_messages = standard_messages[-max_standard:]
    if system_prompt:
        return [system_prompt] + pinned_messages + recent_messages
    return pinned_messages + recent_messages

class VedState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], limit_messages] 
    route_intent: Literal["A", "B", "C", ""] =Field(
        default="",
        description="The workflow route path letter. A=Chat, B=Essay writing loop, C=Python tool scripts."
        )
    current_draft: str
    critique_notes: str
    essay_score: int
    loop_count: int
    mode: str
    saved_memories: list = Field(default_factory=list)