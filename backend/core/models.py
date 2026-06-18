from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


MODELS = {

    # ===== locals via Ollama =====

    "qwen": ChatOllama(
        model="qwen3.5",
        temperature=0
    ),

    "gpt_oss": ChatOllama(
        model="gpt-oss:20b",
        temperature=0
    ),

    "gemma": ChatOllama(
        model="gemma4",
        temperature=0.2
    ),

    "minimax": ChatOllama(
    model="minimax-m2:cloud",
    temperature=0
    ),


    # ===== via API OpenAI =====

    "glm": ChatOpenAI(
        model="glm-5.1",
        base_url="http://localhost:8000/v1",
        api_key="your_api_key",
        temperature=0
    ),

    "nemotron": ChatOpenAI(
        model="nemotron",
        base_url="http://localhost:8001/v1",
        api_key="your_api_key",
        temperature=0
    )
}