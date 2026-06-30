from typing import Dict, Any, List
from langchain_core.messages import AIMessage, HumanMessage
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

    approval_event = None
    approval_state = None
    if config and isinstance(config.get("configurable"), dict):
        approval_event = config["configurable"].get("approval_event")
        approval_state = config["configurable"].get("approval_state")

    # Build a dedicated config instance that completely purges the streaming queue handle
    silent_config = {"configurable": {k: v for k, v in config.get("configurable", {}).items() if k != "token_queue"}} if config else {}
    draft_history: List[Dict[str, Any]] = []
    user_approved_early = False

    while current_state.loop_count <= 2:
        # 1. Execute background generation using the isolated silent config
        gen_res = content_generator_node(current_state, get_llm, silent_config)
        current_state = current_state.model_copy(update=gen_res)

        # 2. Evaluate draft metrics
        eval_res = content_evaluator_node(current_state, get_llm)
        current_state = current_state.model_copy(update=eval_res)

        # 3. Evaluator decided the draft needs more external context. Run a web
        # search once per pipeline invocation and stash the results on state so
        # the NEXT pass (retry) can fold them into its context.
        if current_state.web_search_needed and not current_state.web_search_results:
            try:
                user_messages = [m for m in current_state.messages if isinstance(m, HumanMessage)]
                search_query = user_messages[-1].content if user_messages else current_state.current_draft[:200]
                from graph.tools.web_search import web_search
                results = web_search(search_query, max_results=5)
                current_state = current_state.model_copy(update={
                    "web_search_results": results,
                    "web_search_needed": False,
                })
                if token_queue:
                    if results:
                        token_queue.put(f"\n[System: Web search ran — {len(results)} result(s) attached for next pass.]\n")
                    else:
                        token_queue.put("\n[System: Web search ran — no results.]\n")
            except Exception:
                current_state = current_state.model_copy(update={"web_search_needed": False})

        draft_history.append({
            "score": current_state.content_score,
            "draft": current_state.current_draft,
            "critique": current_state.critique_notes
        })

        # Stream the per-pass draft to the UI so the user can see what they're judging.
        pass_label = f"### Pass {current_state.loop_count + 1}/3 (Score: {current_state.content_score}/100)\n\n"
        if token_queue:
            token_queue.put(f"{pass_label}{current_state.current_draft}\n\n")

        # Human-in-the-loop: ask the user to approve or reject the draft.
        # Skip the request on the final pass (loop_count == 2) — no more button after this.
        is_final_pass = (current_state.loop_count >= 2)
        if approval_event is not None and approval_state is not None and not is_final_pass:
            if token_queue:
                token_queue.put(("approval_request", {
                    "pass": current_state.loop_count + 1,
                    "score": current_state.content_score,
                }))
            approval_event.wait()
            approved = bool(approval_state.get("value"))
            approval_state["value"] = None
            approval_event.clear()
            if approved:
                user_approved_early = True
                break

        current_state = current_state.model_copy(update={"loop_count": current_state.loop_count + 1})

    draft_history.sort(key=lambda x: x["score"], reverse=True)
    best_variant = draft_history[0]

    if user_approved_early:
        # The approved pass IS the chosen best — no need for the redundant final block.
        final_payload = (
            f"### APPROVED OUTPUT (Score: {best_variant['score']}/100)\n\n"
            f"{best_variant['draft']}\n\n"
            f"**Best Draft Critique Analysis:**\n{best_variant['critique']}"
        )
    else:
        final_payload = (
            f"### CHOSEN BEST DRAFT (Score: {best_variant['score']}/100)\n\n"
            f"{best_variant['draft']}\n\n"
            f"**Best Draft Critique Analysis:**\n{best_variant['critique']}"
        )

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