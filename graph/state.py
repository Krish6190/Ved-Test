from typing import TypedDict, Annotated

def cap_message_history(left: list, right: list) -> list:
    """
    Custom reducer that replaces LangGraph's default add_messages.
    It combines old and new messages, but enforces a strict hard-cap 
    of the last 20 messages max in the global graph memory.
    """
    combined = (left or []) + (right or [])
    
    # ENFORCE STATE CAPPING: 
    # If the total stored chat history exceeds 20 items, slice it!
    # This prevents infinite RAM enlargement in the global state.
    if len(combined) > 20:
        return combined[-20:]
        
    return combined

class VedState(TypedDict):
    # Conversation memory: messages are accumulated through the graph run.
    messages: Annotated[list, cap_message_history]

    # Routing decision made by the intent router node.
    route_intent: str

    # Current chatbot mode: standard / turbo / hibernate
    mode: str
