from langchain_nvidia_ai_endpoints import ChatNVIDIA
from backend.core.config import settings

# supports_structured_output=True cho tất cả (trừ "qwen" — giữ riêng cho intake agent,
# schema đơn giản nên không cần ép structured output nghiêm ngặt).
# Ưu tiên các model MoE với active-params thấp để giảm latency cho pipeline tạo đề.

MODELS = {

    "qwen": ChatNVIDIA(
        model="qwen/qwen3-next-80b-a3b-instruct",
        api_key=settings.nim_api_key,
        temperature=0
    ),  # qwen cho intake — bản qwen3-next CÓ supports_structured_output (3.5-122b thì không)

    "gpt_oss": ChatNVIDIA(
        model="openai/gpt-oss-20b",
        api_key=settings.nim_api_key,
        temperature=0
    ),  # giữ nguyên — 20B, nhẹ nhất trong nhóm

    "gemma": ChatNVIDIA(
        model="google/gemma-4-31b-it",
        api_key=settings.nim_api_key,
        temperature=0.2
    ),


    "minimax": ChatNVIDIA(
        model="minimaxai/minimax-m2",
        api_key=settings.nim_api_key,
        temperature=0
    ),  # giữ nguyên — MoE 230B total/10B active, đã đủ nhẹ

    "glm": ChatNVIDIA(
        model="nvidia/nemotron-3-super-120b-a12b",
        api_key=settings.nim_api_key,
        temperature=0
    ),  # thay deepseek-v3.2 (671B/37B active + "think AND tools" mặc định, nặng nhất nhóm cũ)
        # -> Nemotron Super, MoE chỉ 12B active, được NVIDIA công bố throughput cao hơn hẳn
        #    model dense/lớn cùng tầm, confirmed mạnh cho structured/agentic task

    "nemotron": ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        api_key=settings.nim_api_key,
        temperature=0
    ).with_thinking_mode(enabled=False),
    # tắt hẳn reasoning mặc định (nếu không cần cho agent này) để không sinh dư
    # <think>...</think> token — bật lại enabled=True riêng cho agent nào cần giải logic sâu

    "llama31": ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        api_key=settings.nim_api_key,
        temperature=0.3,
        max_completion_tokens=4096,
    ),  # instruct cổ điển, native tool-calling CHẮC (không phải reasoning model)
        # -> đo được 5/5 structured output cho schema generator; KHÔNG lan man -> khỏi retry
}