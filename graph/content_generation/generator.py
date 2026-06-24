from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from ..state import VedState
from graph.rag import rag_db

def content_generator_node(state: VedState, get_llm, config: RunnableConfig) -> Dict[str, Any]:
    """Path B: Generator Stage. Refines draft documents across sliding MMR lookups."""
    llm = get_llm()
    if not llm:
        return {"current_draft": "Error: Local LLM engine is unavailable."}
    
    if hasattr(llm, "temperature"):
        llm.temperature = 0.7
        
    # FIXED: Use explicit safety checks to ensure silent_config blocks streaming leaks
    token_queue = None
    if config and "configurable" in config:
        token_queue = config["configurable"].get("token_queue", None)

    user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    original_request = user_messages[-1].content if user_messages else "Generate asset"

    pass_idx = getattr(state, "loop_count", 0)
    lambda_mult = max(0.2, 0.7 - (pass_idx * 0.25))

    # Perform a hardware-accelerated SIMD matrix vector query
    context_chunks = rag_db.query_similarity(original_request, k=3, lambda_mult=lambda_mult)
    rag_context = "\n---\n".join([c["content"] for c in context_chunks]) if context_chunks else ""

    context_block = ""
    if rag_context:
        context_block = f"\n\n[LOCAL OFFLINE RAG REFERENCE CONTEXT (MMR Balance Slider: {lambda_mult:.2f})]:\n{rag_context}"

    if pass_idx == 0:
        system_prompt = f"You are an expert asset writer. Write a comprehensive long-form draft matching the user prompt.{context_block}"
        prompts = [SystemMessage(content=system_prompt), HumanMessage(content=original_request)]
    else:
        system_prompt = (
            f"You are an editor fixing a draft based on critiques.{context_block}\n"
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
