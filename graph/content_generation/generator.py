from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from ..state import VedState
from graph.rag.mixer import retrieve_context, _format_rag_block, PATH_B_DIVERSITY
from graph.tools.web_search import format_web_results_block

def content_generator_node(state: VedState, get_llm, config: RunnableConfig) -> Dict[str, Any]:
    """Path B: Generator Stage. Refines draft documents across sliding MMR lookups.

    - Pass 0: streams full conversation history to the LLM so it knows what happened before.
    - Pass >= 1: minimal prompt — system + RAG + (last draft + evaluator critique).
    - RAG widens the candidate pool and lowers MMR lambda each pass for diversity.
    """
    llm = get_llm()
    if not llm:
        return {"current_draft": "Error: Local LLM engine is unavailable."}

    if hasattr(llm, "temperature"):
        llm.temperature = 0.7

    token_queue = None
    active_thread_id = None
    if config and isinstance(config.get("configurable"), dict):
        token_queue = config["configurable"].get("token_queue", None)
        active_thread_id = config["configurable"].get("active_thread_id")

    user_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    original_request = user_messages[-1].content if user_messages else "Generate asset"

    pass_idx = getattr(state, "loop_count", 0)
    diversity = PATH_B_DIVERSITY[min(pass_idx, len(PATH_B_DIVERSITY) - 1)]
    k = diversity["k"]
    lambda_mult = diversity["lambda_mult"]

    # On retry, augment the retrieval query with the evaluator's critique so each
    # pass fetches a different angle from the vector store.
    retrieval_query = original_request
    if pass_idx >= 1 and getattr(state, "critique_notes", None):
        retrieval_query = f"{original_request}\n\nCritique:\n{state.critique_notes[:500]}"

    rag_chunks = retrieve_context(
        retrieval_query, active_thread_id, k=k, lambda_mult=lambda_mult,
    )
    rag_block = _format_rag_block(rag_chunks)
    web_block = format_web_results_block(getattr(state, "web_search_results", None) or [])

    if pass_idx == 0:
        # First loop: the generator gets the FULL conversation history so it
        # knows what happened earlier. RAG is prepended as a SystemMessage.
        prompts = list(state.messages)
        prepended = ""
        if rag_block:
            prepended += rag_block
        if web_block:
            prepended += ("\n\n" if prepended else "") + web_block
        if prepended:
            prompts = [SystemMessage(content=prepended)] + prompts
    else:
        # Retry: minimal prompt — system + RAG + web (if any) + (last draft + evaluator critique).
        system = (
            "You are an editor. Use the retrieved context and the critique below "
            "to revise the previous draft. Output only the revised draft."
        )
        if rag_block:
            system = f"{system}\n\n{rag_block}"
        if web_block:
            system = f"{system}\n\n{web_block}"
        critique = (getattr(state, "critique_notes", "") or "").strip()
        human = (
            f"Critique to resolve:\n{critique}\n\n"
            f"Previous draft:\n{state.current_draft}\n\n"
            "Revised draft:"
        )
        prompts = [SystemMessage(content=system), HumanMessage(content=human)]

    full_draft = ""
    for chunk in llm.stream(prompts):
        if chunk.content:
            full_draft += chunk.content
            if token_queue:
                token_queue.put(chunk.content)

    return {"current_draft": full_draft}
