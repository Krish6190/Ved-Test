from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

def limit_messages(left: list, right: list) -> list:
    """Combines messages and keeps only the last 20 (10 complete turns)."""
    combined = add_messages(left, right)
    return combined[-20:]

class VedState(TypedDict):
    # Short-term memory: limited to 20 messages (10 complete turns)
    messages: Annotated[list, limit_messages] 
    # Routing decision made by the intent router node.
    route_intent: str
    mode: str
    saved_memories: list