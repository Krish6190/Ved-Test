from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from ..state import VedState
from .generator import content_generator_node
from .evaluator import content_evaluator_node

def content_pipeline_node(state: VedState, get_llm, config: RunnableConfig) -> dict:
    """Orchestrates Path B evaluation and review loops internally.
    
    Keeps the main nodes file lean by encapsulating generator and evaluator 
    execution states.
    """
    current_state = state
    try:
        token_queue = config["configurable"]["token_queue"]
    except (KeyError, TypeError):
        token_queue = None
    while current_state.loop_count < 2:
        if token_queue:
            token_queue.put(f"\n[Ved: Initiating Content Loop Stage {current_state.loop_count + 1}...]\n")
        
        # 1. Generate / Rewrite Draft Step
        gen_res = content_generator_node(current_state, get_llm, config)
        current_state = current_state.copy(update=gen_res)
        
        # 2. Evaluate Quality Metrics Step
        eval_res = content_evaluator_node(current_state, get_llm)
        current_state = current_state.copy(update=eval_res)
        
        # Divert to Path C if the evaluator flags web search needs
        if current_state.route_intent == "C":
            if token_queue:
                token_queue.put(f"\n[Ved: External validation flagged. Diverting to tool framework...]\n")
            return {
                "current_draft": current_state.current_draft,
                "critique_notes": current_state.critique_notes,
                "content_score": current_state.content_score,
                "loop_count": current_state.loop_count,
                "route_intent": "C"
            }
            
        # Break loop early if score exceeds threshold criteria
        if current_state.content_score >= 80:
            break

    # Build final structural system terminal payload output block 
    final_payload = (
        f"{current_state.current_draft}\n\n"
        f"--- Pipeline Status Report ---\n"
        f"Final Evaluation Score: {current_state.content_score}/100\n"
    )
    if current_state.content_score < 80:
        final_payload += f"Remaining Issues to Resolve:\n{current_state.critique_notes}"

    return {
        "messages": [AIMessage(content=final_payload)],
        "current_draft": current_state.current_draft,
        "critique_notes": current_state.critique_notes,
        "content_score": current_state.content_score,
        "loop_count": 0,  # Reset loop counting states for subsequent system turns
        "route_intent": ""
    }
