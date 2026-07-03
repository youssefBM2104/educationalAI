from langchain_nvidia_ai_endpoints import ChatNVIDIA
from backend.core.config import settings

MODELS = {

    "qwen": ChatNVIDIA(
        model="qwen/qwen3.5-122b-a10b",
        api_key=settings.nim_api_key,
        temperature=0
    ),

    "gpt_oss": ChatNVIDIA(
        model="openai/gpt-oss-20b",
        api_key=settings.nim_api_key,
        temperature=0
    ),

    "gemma": ChatNVIDIA(
        model="google/gemma-4-31b-it",
        api_key=settings.nim_api_key,
        temperature=0.2
    ),

    "minimax": ChatNVIDIA(
    model="minimaxai/minimax-m2.7"  ,
    api_key=settings.nim_api_key,
    temperature=0
    ),


    "glm": ChatNVIDIA(
        model="z-ai/glm-5.1",
        api_key=settings.nim_api_key,
        temperature=0
    ),

    "nemotron": ChatNVIDIA(
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        api_key=settings.nim_api_key,
        temperature=0
    )
}