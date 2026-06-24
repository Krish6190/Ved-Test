# graph/content_generation/pipeline_node.py
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from ..state import VedState
from .generator import content_generator_node
from .evaluator import content_evaluator_node

def content_pipeline_node(state: VedState, get_llm, config: RunnableConfig) -> dict:
    """Orchestrates Path B evaluation and review loops internally.
    Encapsulates 3 clean generation cycles (2 total retries) and handles
    multi-variant score tracking.
    """
    current_state = state
    try:
        token_queue = config["configurable"]["token_queue"]
    except (KeyError, TypeError):
        token_queue = None
        
    draft_history: List[Dict[str, Any]] = []
    
    # loop_count will sequence from 0, to 1, to 2 (Total 3 Generation Attempts)
    while current_state.loop_count <= 2:
        if token_queue:
            token_queue.put(f"\n[System Loop Check: Running Pass {current_state.loop_count + 1} of 3...]\n")
            
        # 1. Generate / Rewrite Draft Step
        gen_res = content_generator_node(current_state, get_llm, config)
        current_state = current_state.copy(update=gen_res)
        
        # 2. Evaluate Quality Metrics Step
        eval_res = content_evaluator_node(current_state, get_llm)
        # Note: evaluator returns updated loop_count = loop_count + 1 internally
        current_state = current_state.copy(update=eval_res)
        
        # Keep an exact record of this variant's execution run
        draft_history.append({
            "score": current_state.content_score,
            "draft": current_state.current_draft,
            "critique": current_state.critique_notes
        })
        
        # Intercept Path C early-exit routing flags immediately
        if current_state.route_intent == "C":
            if token_queue:
                token_queue.put(f"\n[System Action: Web search requested by evaluator. Diverting to Path C...]\n")
            return {
                "current_draft": current_state.current_draft,
                "critique_notes": current_state.critique_notes,
                "content_score": current_state.content_score,
                "loop_count": current_state.loop_count, # Preserve count for tools step
                "route_intent": "C"
            }
            
        # Break out early if the current pass clears the strict quality threshold
        if current_state.content_score >= 80:
            if token_queue:
                token_queue.put(f"\n[System Action: Threshold cleared with a score of {current_state.content_score}/100. Ending cycle early.]\n")
            break

    # Sort historical attempts by score descending to extract the top performers
    draft_history.sort(key=lambda x: x["score"], reverse=True)
    best_variant = draft_history[0]
    
    final_payload = (
        f"### CHOSEN BEST DRAFT (Score: {best_variant['score']}/100)\n\n"
        f"{best_variant['draft']}\n\n"
        f"--- Pipeline Execution Log ---\n"
    )
    
    # If 2 or more generations ran, output the absolute best along with the runner-up score metrics
    if len(draft_history) > 1:
        runner_up = draft_history[1]
        final_payload += (
            f"### RUNNER-UP DRAFT VARIANT (Score: {runner_up['score']}/100)\n\n"
            f"*To view alternatives or run a new asset request, prompt the system below.*\n"
        )
    else:
        final_payload += f"Remaining Improvement Targets from Evaluation:\n{best_variant['critique']}"

    return {
        "messages": [AIMessage(content=final_payload)],
        "current_draft": best_variant["draft"],
        "critique_notes": best_variant["critique"],
        "content_score": best_variant["score"],
        "loop_count": 0, # Reset for next separate incoming system call
        "route_intent": ""
    }