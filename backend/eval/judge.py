import logging
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from backend.core.config import settings

logger = logging.getLogger(__name__)

RAW_MIN, RAW_MAX = 1, 5

# --- Judge backends -----------------------------------------------------------------
#
# "gemini"  — HELD-OUT judge: a different provider from the pipeline (NVIDIA), so the judge
#             never grades output produced by its own family. Use this for any number that
#             is reported. Free tier is only 20 requests/DAY per model, and one 3-question
#             eval costs 12 calls — so it runs dry fast.
#
# "llama31" — NOT held-out: this is the same model the pipeline's own judge_agent runs on.
#             It has already approved these questions once, so its scores are optimistically
#             biased. Fine for iterating on rubrics; do NOT report these numbers.
#
DEFAULT_BACKEND = "gemini"

_BACKENDS = {
    "gemini": {"model": "gemini-2.5-flash", "throttle": 6.5},   # gemini-2.0-flash: free quota = 0
    "llama31": {"model": "meta/llama-3.1-70b-instruct", "throttle": 0.0},
}

_backend = DEFAULT_BACKEND
_judge = None


def set_judge(backend: str) -> None:
    global _backend, _judge
    if backend not in _BACKENDS:
        raise ValueError(f"unknown judge backend {backend!r}; choose from {list(_BACKENDS)}")
    _backend, _judge = backend, None


def judge_name() -> str:
    tag = "" if _backend == "gemini" else "  [NOT held-out — do not report]"
    return f"{_BACKENDS[_backend]['model']}{tag}"


def _throttle_seconds() -> float:
    return _BACKENDS[_backend]["throttle"]


def get_judge():
    global _judge
    if _judge is None:
        if _backend == "llama31":
            from backend.core.models import MODELS
            _judge = MODELS["llama31"]
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            if not settings.google_api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
                    "and add GOOGLE_API_KEY=... to .env"
                )
            _judge = ChatGoogleGenerativeAI(
                model=_BACKENDS["gemini"]["model"],
                google_api_key=settings.google_api_key,
                temperature=0,
            )
    return _judge


class Verdict(BaseModel):
    reason: str = Field(description="Concise justification, referring to the evaluation steps")
    score: int = Field(ge=RAW_MIN, le=RAW_MAX, description=f"Score from {RAW_MIN} (worst) to {RAW_MAX} (best)")


@dataclass
class Result:
    name: str
    score: float          # normalised 0-1
    reason: str
    passed: bool


@dataclass
class GEval:
    """One rubric. `fields` names which parts of the test case the judge is allowed to see —
    withholding a field is a deliberate design choice (e.g. the naturalness metric must NOT
    see the knowledge graph, otherwise it cannot tell that a question is only answerable
    by looking at the graph)."""
    name: str
    criteria: str
    evaluation_steps: list[str]
    fields: list[str]
    threshold: float = 0.7
    _last_call: float = field(default=0.0, repr=False)

    def _prompt(self) -> str:
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(self.evaluation_steps, 1))
        return f"""## Role
You are a strict evaluator of automatically generated exam questions.

## Criterion — {self.name}
{self.criteria}

## Evaluation steps
Work through these in order, then score.

{steps}

## Scoring
- **{RAW_MAX}** — fully satisfies the criterion.
- **3** — partially satisfies it; noticeable problems.
- **{RAW_MIN}** — clearly violates the criterion.

Judge ONLY this criterion. Ignore issues that belong to other criteria.
Give the reason first, then the score.
""".strip()

    def measure(self, case: dict) -> Result:
        shown = {k: v for k, v in case.items() if k in self.fields}
        body = "\n\n".join(f"### {k}\n{v}" for k, v in shown.items())

        gap = _throttle_seconds()
        elapsed = time.monotonic() - self._last_call
        if self._last_call and elapsed < gap:
            time.sleep(gap - elapsed)

        verdict = get_judge().with_structured_output(Verdict).invoke([
            SystemMessage(content=self._prompt()),
            HumanMessage(content=body),
        ])
        self._last_call = time.monotonic()

        score = (verdict.score - RAW_MIN) / (RAW_MAX - RAW_MIN)   # -> 0..1
        logger.info("GEval[%s]: %.2f", self.name, score)
        return Result(
            name=self.name,
            score=score,
            reason=verdict.reason,
            passed=score >= self.threshold,
        )
