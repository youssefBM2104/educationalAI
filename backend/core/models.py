from langchain_nvidia_ai_endpoints import ChatNVIDIA
from backend.core.config import settings

MODELS = {

    "qwen": ChatNVIDIA(
        model="qwen3.5",
        api_key=settings.nim_api_key,
        temperature=0
    ),

    "gpt_oss": ChatNVIDIA(
        model="gpt-oss:20b",
        api_key=settings.nim_api_key,
        temperature=0
    ),

    "gemma": ChatNVIDIA(
        model="gemma4",
        api_key=settings.nim_api_key,
        temperature=0.2
    ),

    "minimax": ChatNVIDIA(
    model="minimax-m2:cloud",
    api_key=settings.nim_api_key,
    temperature=0
    ),


    "glm": ChatNVIDIA(
        model="glm-5.1",
        api_key=settings.nim_api_key,
        temperature=0
    ),

    "nemotron": ChatNVIDIA(
        model="nemotron",
        api_key=settings.nim_api_key,
        temperature=0
    )
}