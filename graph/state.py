from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, Field

def limit_messages(left: Sequence[BaseMessage], right: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Combines message history arrays. Preserves the top system prompt and protects long-term memory pins."""
    combined = add_messages(left, right)
    system_prompt = None
    pinned_messages = []
    normal_messages = []
    
    for msg in combined:
        if isinstance(msg, SystemMessage) and system_prompt is None:
            system_prompt = msg
            continue
        if getattr(msg, "additional_kwargs", {}).get("pinned", False):
            pinned_messages.append(msg)
        else:
            normal_messages.append(msg)
            
    if len(pinned_messages) > 20:
        pinned_messages = pinned_messages[-20:]
        
    max_normal_slots = 39 - len(pinned_messages)
    recent_messages = normal_messages[-max_normal_slots:] if max_normal_slots > 0 else []
    
    if system_prompt:
        return [system_prompt] + pinned_messages + recent_messages
    return pinned_messages + recent_messages

class VedState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], limit_messages]
    route_intent: Literal["A", "B", "C", ""] = Field(default="")
    current_draft: str = Field(default="")
    critique_notes: str = Field(default="")
    content_score: int = Field(default=0)
    loop_count: int = Field(default=0)
    mode: str = Field(default="standard")
    saved_memories: list = Field(default_factory=list)
    # Web search signal (set by Path B evaluator when more external context is needed)
    # and the results fetched in response. Path A reads/writes these too via chat_node.
    web_search_needed: bool = Field(default=False)
    web_search_results: list = Field(default_factory=list)
    self_healing: bool = Field(default=False)
