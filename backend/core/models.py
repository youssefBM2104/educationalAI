from langchain_nvidia_ai_endpoints import ChatNVIDIA
from backend.core.config import settings



MODELS = {

    "qwen": ChatNVIDIA(
        model="qwen/qwen3-next-80b-a3b-instruct",
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
        temperature=0.3
    ),


    "minimax": ChatNVIDIA(
        model="minimaxai/minimax-m2",
        api_key=settings.nim_api_key,
        temperature=0
    ), 

    "glm": ChatNVIDIA(
        model="nvidia/nemotron-3-super-120b-a12b",
        api_key=settings.nim_api_key,
        temperature=0
    ), 

    "nemotron": ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        api_key=settings.nim_api_key,
        temperature=0
    ).with_thinking_mode(enabled=False),
    

    "llama31": ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        api_key=settings.nim_api_key,
        temperature=0,
        max_completion_tokens=4096,
    ),  
}