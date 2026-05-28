"""Quick test: 1 LLM call to NIM to see if quota/key/network works."""
from dotenv import load_dotenv
load_dotenv()
import os

key = os.getenv("NIM_API_KEY")
print(f"NIM_API_KEY loaded: {bool(key)} (len={len(key or '')})")
if not key:
    raise SystemExit("NIM_API_KEY not in .env — fix that first.")

from langchain_nvidia_ai_endpoints import ChatNVIDIA
llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", api_key=key, temperature=0)

print("Calling NIM (1 single request)...")
try:
    result = llm.invoke("Reply with exactly: Hello world!")
    print(f"\n✅ SUCCESS — NIM responded: {result.content!r}")
    print("\n→ API key works. NIM available. Vấn đề KG là CONCURRENT limit.")
    print("→ Fix: apply throttle (semaphore + retry) như đã propose.")
except Exception as e:
    msg = str(e)
    print(f"\n❌ FAILED: {msg}")
    if "429" in msg:
        print("→ Daily quota cạn. Đợi reset 0h UTC, hoặc đổi sang Ollama.")
    elif "401" in msg:
        print("→ API key sai hoặc hết hạn. Check NIM_API_KEY trong .env.")
    else:
        print("→ Lỗi khác — paste cho Claude xem.")
