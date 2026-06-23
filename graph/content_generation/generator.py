from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from ..state import VedState

def content_generator_node(state: VedState, get_llm, config: RunnableConfig) -> Dict[str, Any]:
    """Path B: Generator Stage. 
    Generates a new long-form draft asset or refines an existing one using 
    actionable critique feedback metrics.
    """
    llm = get_llm()
    if not llm:
        return {"current_draft": "Error: Local LLM engine is unavailable."}
    if hasattr(llm, "temperature"):
        llm.temperature = 0.7
    try:
        token_queue = config["configurable"]["token_queue"]
    except (KeyError, TypeError):
        token_queue = None
    user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    original_request = user_messages[-1].content if user_messages else "Generate asset"

    rag_context = ""
    if state.saved_memories:
        rag_context = "\n\n[SUPPLEMENTAL RESEARCH CONTEXT]:\n" + "\n".join([str(m) for m in state.saved_memories])

    if state.loop_count == 0:
        system_prompt = f"You are an expert asset writer. Write a comprehensive long-form draft matching the user prompt.{rag_context}"
        prompts = [SystemMessage(content=system_prompt), HumanMessage(content=original_request)]
    else:
        system_prompt = (
            f"You are an editor fixing a draft based on critiques.{rag_context}\n"
            f"Critique notes to resolve:\n{state.critique_notes}"
        )
        prompts = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original Request: {original_request}\n\nPrevious Draft:\n{state.current_draft}")
        ]

    full_draft = ""
    for chunk in llm.stream(prompts):
        if chunk.content:
            full_draft += chunk.content
            if token_queue:
                token_queue.put(chunk.content)

    return {"current_draft": full_draft}