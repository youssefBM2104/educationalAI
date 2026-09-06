"""
LLM-as-judge for exam evaluation, backed by DeepEval's G-Eval.

Why DeepEval (not the earlier hand-rolled judge): for the paper we want a standard, citable
metric implementation. DeepEval's `GEval` is the canonical G-Eval (Liu et al., 2023) — form-
filling with evaluation steps, continuous 0-1 score. We keep our own thin `GEval` wrapper so the
rest of the eval code (metrics_exam.py, run_exam_eval.py) is unchanged: same constructor
(name/criteria/evaluation_steps/fields), same `.measure(case) -> Result`.

Held-out judge is preserved. DeepEval defaults to OpenAI; we instead wrap OUR judge models
(gemini / gpt-4o / llama-3.1) in a DeepEvalBaseLLM so the judge can be a DIFFERENT provider from
the generator (gpt-5) — the methodological point of a held-out judge. Pick the backend with
`set_judge(...)`; `judge_name()` tags how held-out it is.

Field visibility is still per-metric: a metric's `fields` decide which DeepEval test-case slots are
populated AND which `evaluation_params` the judge sees, so withholding the KG from the naturalness
judge still works (it maps to CONTEXT, which naturalness never requests).
"""
import logging
import os
import time
from dataclasses import dataclass, field

from backend.core.config import settings

# Keep DeepEval quiet and offline-friendly (no telemetry / update pings during a run).
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")

from deepeval.metrics import GEval as _DeepEvalGEval          # noqa: E402
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402
from deepeval.models.base_model import DeepEvalBaseLLM        # noqa: E402

logger = logging.getLogger(__name__)


# --- Judge backends -----------------------------------------------------------------
#
# "gemini"  — HELD-OUT judge: a different provider from the generator, so it never grades output
#             from its own family. Cleanest number to report. Free tier is only ~20 requests/DAY,
#             so it runs dry fast on a multi-metric batch.
# "gpt4o"   — a DIFFERENT model from the generator (gpt-5) but the SAME provider (OpenAI). No free-
#             tier cap, scores a whole batch in one go. Reasonable to report; note the shared provider.
# "llama31" — the pipeline's own judge_agent model: already approved these questions, so biased.
#             Use only to iterate on rubrics, never to report.
#
DEFAULT_BACKEND = "gpt4o"

_BACKENDS = {
    "gemini": {"model": "gemini-2.5-flash", "throttle": 6.5},
    "llama31": {"model": "meta/llama-3.1-70b-instruct", "throttle": 0.0},
    "gpt4o": {"model": "gpt-4o", "throttle": 0.0},
}

# Backends served through our own MODELS registry (NVIDIA NIM / OpenAI) vs. the Gemini SDK.
_REGISTRY_MODELS = {"llama31": "llama31", "gpt4o": "gpt-4o"}

_backend = DEFAULT_BACKEND
_judge_model = None          # DeepEvalBaseLLM wrapper, rebuilt on set_judge


def set_judge(backend: str) -> None:
    global _backend, _judge_model
    if backend not in _BACKENDS:
        raise ValueError(f"unknown judge backend {backend!r}; choose from {list(_BACKENDS)}")
    _backend, _judge_model = backend, None


def judge_name() -> str:
    if _backend == "gemini":
        tag = ""                                              # cleanest held-out judge
    elif _backend == "gpt4o":
        tag = "  [different model, but same provider as generator]"
    else:  # llama31 — the pipeline's own judge model
        tag = "  [NOT held-out — do not report]"
    return f"{_BACKENDS[_backend]['model']}{tag}"


def _throttle_seconds() -> float:
    return _BACKENDS[_backend]["throttle"]


def _build_langchain_judge():
    """The underlying LangChain chat model for the active backend."""
    if _backend in _REGISTRY_MODELS:
        from backend.core.models import MODELS
        return MODELS[_REGISTRY_MODELS[_backend]]
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
            "and add GOOGLE_API_KEY=... to .env"
        )
    return ChatGoogleGenerativeAI(
        model=_BACKENDS["gemini"]["model"],
        google_api_key=settings.google_api_key,
        temperature=0,
    )


# --- DeepEval judge model wrapping a LangChain chat model ----------------------------

class _LangChainJudge(DeepEvalBaseLLM):
    """Adapts any LangChain chat model to DeepEval's judge interface. We do NOT advertise log-prob
    support, so GEval takes its schema path (generate_with_schema -> generate(prompt, schema)),
    which returns a validated schema instance via LangChain's with_structured_output."""

    def __init__(self, model, name: str):
        self._model = model
        self._name = name
        # NVIDIA NIM needs tool-calling for structured output; OpenAI/Gemini accept the default.
        self._structured_method = (
            "function_calling" if "NVIDIA" in type(model).__name__ else None
        )
        super().__init__(model_name=name)

    def load_model(self):
        return self._model

    def get_model_name(self) -> str:
        return self._name

    def _structured(self, schema):
        if self._structured_method:
            return self._model.with_structured_output(schema, method=self._structured_method)
        return self._model.with_structured_output(schema)

    def generate(self, prompt, schema=None, **kwargs):
        if schema is not None:
            return self._structured(schema).invoke(prompt)
        return self._model.invoke(prompt).content

    async def a_generate(self, prompt, schema=None, **kwargs):
        if schema is not None:
            return await self._structured(schema).ainvoke(prompt)
        return (await self._model.ainvoke(prompt)).content


def _judge():
    global _judge_model
    if _judge_model is None:
        _judge_model = _LangChainJudge(_build_langchain_judge(), _BACKENDS[_backend]["model"])
    return _judge_model


# --- Field -> DeepEval test-case slot mapping ---------------------------------------
#
# Our cases carry named fields; DeepEval has fixed slots. We map so that each metric shows the
# judge only the fields it asked for (visibility is per-metric). kg_relations -> CONTEXT is why a
# metric that omits kg_relations never leaks the graph to the judge.
_INPUT_FIELDS = ("query", "difficulty", "question", "options")      # -> INPUT
_OUTPUT_FIELDS = ("correct_answer", "explanation", "model_answer")  # -> ACTUAL_OUTPUT
# "chunks" -> RETRIEVAL_CONTEXT ; "kg_relations" -> CONTEXT


@dataclass
class Result:
    name: str
    score: float          # 0-1
    reason: str
    passed: bool


@dataclass
class GEval:
    """One rubric. `fields` names which parts of the test case the judge may see — withholding a
    field is deliberate (the naturalness metric must NOT see the KG, else it cannot tell a question
    is only answerable by looking at the graph)."""
    name: str
    criteria: str
    evaluation_steps: list[str]
    fields: list[str]
    threshold: float = 0.7
    _last_call: float = field(default=0.0, repr=False)

    def _build_test_case(self, case: dict) -> tuple[LLMTestCase, list]:
        shown = {k: v for k, v in case.items() if k in self.fields}

        def _join(keys):
            parts = [f"{k}:\n{shown[k]}" for k in keys if k in shown and shown[k] not in (None, "")]
            return "\n\n".join(parts)

        input_text = _join(_INPUT_FIELDS)
        output_text = _join(_OUTPUT_FIELDS)

        params: list = []
        if input_text:
            params.append(SingleTurnParams.INPUT)
        if output_text:
            params.append(SingleTurnParams.ACTUAL_OUTPUT)

        retrieval = None
        if "chunks" in shown and shown["chunks"]:
            retrieval = [str(shown["chunks"])]
            params.append(SingleTurnParams.RETRIEVAL_CONTEXT)

        context = None
        if "kg_relations" in shown and shown["kg_relations"]:
            context = [str(shown["kg_relations"])]
            params.append(SingleTurnParams.CONTEXT)

        tc = LLMTestCase(
            input=input_text or "(no input)",
            actual_output=output_text or "(not evaluated)",   # LLMTestCase requires this slot
            retrieval_context=retrieval,
            context=context,
        )
        return tc, params

    def measure(self, case: dict) -> Result:
        tc, params = self._build_test_case(case)

        metric = _DeepEvalGEval(
            name=self.name,
            evaluation_steps=self.evaluation_steps,
            evaluation_params=params,
            model=_judge(),
            threshold=self.threshold,
            async_mode=False,      # sync path -> generate_with_schema, no nested event loop
            verbose_mode=False,
        )

        gap = _throttle_seconds()
        elapsed = time.monotonic() - self._last_call
        if self._last_call and elapsed < gap:
            time.sleep(gap - elapsed)

        try:
            metric.measure(tc, _show_indicator=False)
        except TypeError:
            metric.measure(tc)
        self._last_call = time.monotonic()

        score = float(metric.score or 0.0)
        logger.info("GEval[%s]: %.2f", self.name, score)
        return Result(
            name=self.name,
            score=score,
            reason=metric.reason or "",
            passed=score >= self.threshold,
        )
