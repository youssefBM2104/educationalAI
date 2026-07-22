import logging

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.models import MODELS
from backend.agents.state import TutoringState

logger = logging.getLogger(__name__)

llm = MODELS["qwen"]

CONFIDENCE_THRESHOLD = 0.5

CLASSIFY_PROMPT = """
## Role
You classify the cognitive complexity of a student's question on **Bloom's Taxonomy**.

## Levels
1. **Remember** — recall facts
2. **Understand** — explain ideas
3. **Apply** — use knowledge in a new situation
4. **Analyze** — break down, find relationships
5. **Evaluate** — justify, judge
6. **Create** — produce something new

## Output
- `level`: the Bloom level (1-6)
- `confidence`: how sure you are (0-1)
""".strip()


class BloomClassification(BaseModel):
    level: int = Field(ge=1, le=6, description="Bloom level 1..6")
    confidence: float = Field(ge=0.0, le=1.0)


def bloom_classifier(state: TutoringState) -> dict:
    res = llm.with_structured_output(BloomClassification).invoke([
        SystemMessage(content=CLASSIFY_PROMPT),
        HumanMessage(content=state["query"]),
    ])
    level = res.level
    if res.confidence < CONFIDENCE_THRESHOLD:
        level = min(level + 1, 6)  # low confidence -> prefer higher level
    logger.info("Bloom: level=%d confidence=%.2f -> %d", res.level, res.confidence, level)
    return {"bloom_level": level}
