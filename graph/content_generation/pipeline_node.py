from typing import Dict, Any, List
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
    draft_history: List[Dict[str, Any]] = []
    while current_state.loop_count < 2:
        if token_queue:
            token_queue.put(f"\n[Ved: Initiating Content Loop Stage {current_state.loop_count + 1}...]\n")
        # 1. Generate / Rewrite Draft Step
        gen_res = content_generator_node(current_state, get_llm, config)
        current_state = current_state.copy(update=gen_res)
        # 2. Evaluate Quality Metrics Step
        eval_res = content_evaluator_node(current_state, get_llm)
        current_state = current_state.copy(update=eval_res)
        draft_history.append({
            "score": current_state.content_score,
            "draft": current_state.current_draft,
            "critique": current_state.critique_notes
        })
        # Divert to Path C if the evaluator flags web search needs
        if current_state.route_intent == "C":
            if token_queue:
                token_queue.put(f"\n[Ved: External validation flagged. Diverting to tool framework...]\n")
            return {
                "current_draft": current_state.current_draft,
                "critique_notes": current_state.critique_notes,
                "content_score": current_state.content_score,
                "loop_count": current_state.loop_count+1,
                "route_intent": "C"
            }
            
        # Break loop early if score exceeds threshold criteria
        if current_state.content_score >= 80:
            if token_queue:
                token_queue.put(f"\n[Ved: Quality threshold met with a score of {current_state.content_score}/100.]\n")
            break
        current_state = current_state.copy(update={"loop_count": current_state.loop_count + 1})
    draft_history.sort(key=lambda x: x["score"], reverse=True)
    best_variant = draft_history[0]
    final_payload = (
        f"### CHOSEN BEST DRAFT (Score: {best_variant['score']}/100)\n\n"
        f"{best_variant['draft']}\n\n"
        f"--- Pipeline Quality Execution Log ---\n"
    )
    if len(draft_history) > 1:
        runner_up = draft_history[1]
        final_payload += (
            f"### RUNNER-UP ALTERNATIVE DRAFT (Score: {runner_up['score']}/100)\n\n"
            f"```markdown\n{runner_up['draft']}\n```\n"
        )
    else:
        final_payload += f"Remaining Improvement Targets:\n{best_variant['critique']}"

    return {
        "messages": [AIMessage(content=final_payload)],
        "current_draft": best_variant["draft"],
        "critique_notes": best_variant["critique"],
        "content_score": best_variant["score"],
        "loop_count": 0, 
        "route_intent": ""
    }