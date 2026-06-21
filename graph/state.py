from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages


class VedState(TypedDict):
    # Conversation memory: messages are accumulated through the graph run.
    messages: Annotated[list, add_messages]

    # Routing decision made by the intent router node.
    route_intent: str

    # Current chatbot mode: standard / turbo / hibernate
    mode: str
    saved_memories: list