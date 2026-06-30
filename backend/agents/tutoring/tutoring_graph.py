import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from backend.agents.state import TutoringState
from backend.agents.tutoring.classifier_agent import bloom_classifier
from backend.agents.tutoring.qa_generator_agent import generate_subquestion
from backend.agents.tutoring.evaluator_agent import response_evaluator
from backend.agents.tutoring.bridge_agent import bridge_generator
from backend.agents.tutoring.diagnoser_agent import diagnoser
from backend.agents.tutoring.answer_agent import answer_node, teach_explanation

logger = logging.getLogger(__name__)

_app = None
FAIL_CAP = 3


# --- Nodes ---

def gen_subq_node(state: TutoringState) -> dict:
    n = state["bloom_level"]
    level = state.get("current_level")
    if level is None:
        level = n - 1  # Socratic loop starts one level below N

    sub_q, expected = generate_subquestion(state, level)
    teach = state.get("teach_note")
    message = (teach + "\n\n" + sub_q) if teach else sub_q

    return {
        "current_level": level,
        "sub_question": sub_q,
        "expected_answer": expected,
        "phase": "await_sub_answer",
        "tutor_message": message,
        "teach_note": None,
        "fail_streak": state.get("fail_streak") or {},
    }


def scaffold_node(state: TutoringState) -> dict:
    level = state["current_level"]
    target = state["bloom_level"] - 1
    streak = dict(state.get("fail_streak") or {})
    streak[level] = streak.get(level, 0) + 1
    update = {"fail_streak": streak}

    if streak[level] >= FAIL_CAP:
        # safety cap: teach this level directly, then climb
        update["teach_note"] = teach_explanation(state)
        update["current_level"] = min(level + 1, target)
    elif level > 1:
        update["current_level"] = level - 1  # descend
    else:
        # fail at level 1: teach directly, then climb
        update["teach_note"] = teach_explanation(state)
        update["current_level"] = min(2, target)

    return update


def climb_node(state: TutoringState) -> dict:
    level = state["current_level"]
    target = state["bloom_level"] - 1
    if level >= target:
        return {"next_step": "bridge"}      # passed N-1 -> bridge step
    return {"current_level": level + 1, "next_step": "ask"}  # climb back, re-ask


# --- Routing ---

def route_entry(state: TutoringState) -> str:
    phase = state.get("phase")
    if phase == "await_sub_answer":
        return "evaluate"
    if phase == "await_bridge":
        return "diagnose"
    return "classify"


def route_after_classify(state: TutoringState) -> str:
    return "answer" if state["bloom_level"] <= 2 else "gen_subq"


def route_after_evaluate(state: TutoringState) -> str:
    return "climb" if state.get("eval_result") == "pass" else "scaffold"


def route_after_climb(state: TutoringState) -> str:
    return "gen_bridge" if state.get("next_step") == "bridge" else "gen_subq"


# --- Graph (turn-based: one student turn per invoke; state persists via checkpointer) ---

def get_tutoring_graph():
    global _app
    if _app is not None:
        return _app

    g = StateGraph(TutoringState)

    g.add_node("classify", bloom_classifier)
    g.add_node("gen_subq", gen_subq_node)
    g.add_node("evaluate", response_evaluator)
    g.add_node("scaffold", scaffold_node)
    g.add_node("climb", climb_node)
    g.add_node("gen_bridge", bridge_generator)
    g.add_node("diagnose", diagnoser)
    g.add_node("answer", answer_node)

    g.add_conditional_edges(START, route_entry, {
        "classify": "classify",
        "evaluate": "evaluate",
        "diagnose": "diagnose",
    })
    g.add_conditional_edges("classify", route_after_classify, {
        "answer": "answer",
        "gen_subq": "gen_subq",
    })
    g.add_edge("gen_subq", END)                      # ask sub-question, wait for next turn

    g.add_conditional_edges("evaluate", route_after_evaluate, {
        "climb": "climb",
        "scaffold": "scaffold",
    })
    g.add_edge("scaffold", "gen_subq")               # descend / teach -> re-ask
    g.add_conditional_edges("climb", route_after_climb, {
        "gen_subq": "gen_subq",
        "gen_bridge": "gen_bridge",
    })
    g.add_edge("gen_bridge", END)                    # ask bridge, wait for next turn

    g.add_edge("diagnose", "answer")
    g.add_edge("answer", END)

    _app = g.compile(checkpointer=MemorySaver())
    return _app
