from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from ..state import VedState
from .generator import content_generator_node
from .evaluator import content_evaluator_node

def content_pipeline_node(state: VedState, get_llm, config: RunnableConfig) -> dict:
    """Orchestrates Path B evaluation loops silently, ensuring clean output delivery."""
    current_state = state
    try:
        token_queue = config["configurable"]["token_queue"]
    except (KeyError, TypeError):
        token_queue = None
        
    # Build a dedicated config instance that completely purges the streaming queue handle
    silent_config = {"configurable": {k: v for k, v in config.get("configurable", {}).items() if k != "token_queue"}} if config else {}
    draft_history: List[Dict[str, Any]] = []
    
    while current_state.loop_count <= 2:
        if token_queue:
            token_queue.put(f"\n[System Loop Check: Running Pass {current_state.loop_count + 1} of 3...]\n")
            
        # 1. Execute background generation using the isolated silent config
        gen_res = content_generator_node(current_state, get_llm, silent_config)
        current_state = current_state.model_copy(update=gen_res)
        
        # 2. Evaluate draft metrics
        eval_res = content_evaluator_node(current_state, get_llm)
        current_state = current_state.model_copy(update=eval_res)
        
        draft_history.append({
            "score": current_state.content_score,
            "draft": current_state.current_draft,
            "critique": current_state.critique_notes
        })
        
        if current_state.route_intent == "C":
            if token_queue:
                token_queue.put(f"[Tool Interlock]: Web search flagged. Re-routing to diverse local RAG...\n")
            current_state = current_state.model_copy(update={"route_intent": "B"})
            
        if current_state.content_score >= 80:
            if token_queue:
                token_queue.put(f"[System Match]: Quality criteria met at {current_state.content_score}/100.\n\n")
            break

        current_state = current_state.model_copy(update={"loop_count": current_state.loop_count + 1})

    draft_history.sort(key=lambda x: x["score"], reverse=True)
    best_variant = draft_history[0]
    
    final_payload = (
        f"### CHOSEN BEST DRAFT (Score: {best_variant['score']}/100)\n\n"
        f"{best_variant['draft']}\n\n"
        f"**Best Draft Critique Analysis:**\n{best_variant['critique']}\n\n"
        f"--- Alternative Pipeline Outputs ---\n\n"
    )
    
    if len(draft_history) > 1:
        runner_up = draft_history[1]
        final_payload += (
            f"### RUNNER-UP DRAFT VARIANT (Score: {runner_up['score']}/100)\n\n"
            f"{runner_up['draft']}\n"
        )
    else:
        final_payload += f"Remaining Improvement Targets from Evaluation:\n{best_variant['critique']}\n"

    # Stream out the finalized draft structure to the user interface render view
    if token_queue:
            token_queue.put(final_payload)

    return {
        "messages": [AIMessage(content=final_payload)],
        "current_draft": best_variant["draft"],
        "critique_notes": best_variant["critique"],
        "content_score": best_variant["score"],
        "loop_count": 0, 
        "route_intent": ""
    }