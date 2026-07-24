from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from backend.core.config import settings


def _openai(model: str, reasoning_effort: str = "low") -> ChatOpenAI:
    # GPT-5 models "think" before answering, and those hidden reasoning tokens dominate cost +
    # latency (a first run spent 26k reasoning tokens on a 3-question exam). These are extraction
    # / judging tasks, not deep problem-solving, so cap reasoning_effort per tier below.
    # Temperature is left unset — GPT-5 reasoning models reject a non-default value.
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        reasoning_effort=reasoning_effort,
    )


MODELS = {

    "qwen": ChatNVIDIA(
        model="qwen/qwen3-next-80b-a3b-instruct",
        api_key=settings.nim_api_key,
        temperature=0
    ),  

    "gemma": ChatNVIDIA(
        model="google/gemma-4-31b-it",
        api_key=settings.nim_api_key,
        temperature=0.2
    ),

    "llama31": ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        api_key=settings.nim_api_key,
        temperature=0,
        max_completion_tokens=4096,
    ),
    
    "gpt-5-nano": _openai("gpt-5-nano", reasoning_effort="minimal"),  # intake/mindmap — trivial
    "gpt-5-mini": _openai("gpt-5-mini", reasoning_effort="minimal"),
    "gpt-5":      _openai("gpt-5", reasoning_effort="low"),           # gen/solver — some path reasoning
    "gpt-5.1":    _openai("gpt-5.1", reasoning_effort="low"),         # judge

    # Non-reasoning model with a tunable temperature — alternative for the generator, where natural
    # creative phrasing matters more than deep path reasoning (and there are no hidden reasoning
    # tokens). Switch generator_agent's llm to MODELS["gpt-4o"], then compare eval scores vs gpt-5:
    # expect Naturalness up, watch Groundedness/Difficulty for a drop.
    "gpt-4o": ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key, temperature=0),
}