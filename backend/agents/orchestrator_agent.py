import logging

from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import AgentState

logger = logging.getLogger(__name__)

llm = MODELS["qwen"]

VALID_INTENTS = ("exam", "slides", "mindmap", "summary", "tutoring")
DEFAULT_INTENT = "tutoring"

ROUTING_PROMPT = """
You are an orchestrator in an educational AI system.
Classify the user's request into exactly one intent:

- "exam"     : generate exam questions, a quiz, a test
- "slides"   : build lecture slides / a slide deck / a presentation
- "mindmap"  : build a mind map / concept map
- "summary"  : summarize content into a written summary
- "tutoring" : explanations, help understanding, or a step-by-step learning path

Respond with ONLY one word: exam, slides, mindmap, summary, or tutoring.
""".strip()


def orchestrator_agent(state: AgentState) -> dict:
    """Node: classify user query intent and write to state."""
    response = llm.invoke([
        SystemMessage(content=ROUTING_PROMPT),
        HumanMessage(content=state["query"]),
    ])
    intent = response.content.strip().lower()

    if intent not in VALID_INTENTS:
        logger.warning("Invalid intent %r → falling back to %s", intent, DEFAULT_INTENT)
        intent = DEFAULT_INTENT

    logger.info("Routed query → intent=%s", intent)
    return {"intent": intent}


def route_intent(state: AgentState) -> str:
    """Conditional edge: return intent string for graph routing."""
    return state["intent"]
