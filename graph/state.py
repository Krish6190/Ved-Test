from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

def cap_message_history(left: list, right: list) -> list:
    """
    Safely merges message layers using LangGraph's native engine 
    to preserve hidden object properties and tools metadata.
    """
    # Use LangGraph's built-in tool to append messages without destroying attributes
    merged = add_messages(left or [], right or [])
    
    # If the history gets too long, keep the last 20 items using safe list tracking
    if len(merged) > 20:
        return merged[-20:]
    return merged

class VedState(TypedDict):
    # Conversation memory: messages are accumulated through the graph run.
    messages: Annotated[list, cap_message_history]

    # Routing decision made by the intent router node.
    route_intent: str

    # Current chatbot mode: standard / turbo / hibernate
    mode: str
    saved_memories: list