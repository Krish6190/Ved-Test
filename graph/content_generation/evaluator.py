# graph/content_generation/evaluator.py
import json
import re
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from ..state import VedState

class EvaluationSchema(BaseModel):
    score: int = Field(description="Quality score from 1 to 100.", ge=1, le=100)
    critique: str = Field(description="Actionable details on what exactly needs fixing.")
    web_search_needed: bool = Field(description="True if external factual data or scraping is needed.")

def content_evaluator_node(state: VedState, get_llm) -> Dict[str, Any]:
    """Path B: Evaluator Stage.
    Scores the draft asset under strict temperature constraints and determines 
    if research or rewrite pathways should execute.
    """
    llm = get_llm()
    if not llm:
        return {"content_score": 0, "critique_notes": "LLM offline", "route_intent": "B"}

    if hasattr(llm, "temperature"):
        llm.temperature = 0.0

    eval_prompt = (
        "You are an expert document inspector. Analyze the provided draft text for clarity, structure, and correctness.\n\n"
        f"DRAFT DOCUMENT:\n{state.current_draft}\n\n"
        "GRADING SCALARS (CRITICAL):\n"
        "- Score 80-100: Coherent text answering the request, even if minor phrasing tweaks are possible.\n"
        "- Score 50-79: Text has structural content but missed length constraints or style guides slightly.\n"
        "- Score 1-49: Text is corrupted, empty, or completely off-topic.\n\n"
        "CRITICAL FORMAT RULE:\n"
        "Return EXACTLY a raw JSON block with no other text, markdown wrapper formatting, or explanations.\n"
        'JSON Template: {"score": int, "critique": "string description", "web_search_needed": bool}'
    )

    try:
        structured_llm = llm.with_structured_output(EvaluationSchema)
        res = structured_llm.invoke([SystemMessage(content=eval_prompt)])
        score, critique, search = res.score, res.critique, res.web_search_needed
    except Exception:
        fallback_prompt = (
            f"{eval_prompt}\n\nReturn EXACTLY a raw JSON block with no other text:\n"
            '{"score": int, "critique": "string", "web_search_needed": bool}'
        )
        try:
            raw = llm.invoke([SystemMessage(content=fallback_prompt)])
            match = re.search(r"\{.*\}", raw.content, re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
            score = int(data.get("score", 50))
            critique = str(data.get("critique", "Fallback parse error occurred."))
            search = bool(data.get("web_search_needed", False))
        except Exception:
            score = 75 if state.loop_count >= 1 else 50
            critique = "Critique generation failed during fallback parsing parsing protocols."
            search = False

    # Route to Path C if external validation is explicitly flagged by the schema
    next_route = "C" if search else "B"
    
    return {
        "content_score": score,
        "critique_notes": critique,
        "route_intent": next_route
        # FIXED: Removed internal loop_count increment to prevent dual-increment tracking bugs
    }
