from backend.local_ai.model_manager import model_manager

SYSTEM_PROMPT = """You are a helpful educational assistant.
Answer the student's question clearly and concisely based only on the context provided.
If the answer is not in the context, say so honestly — do not invent information.
Always respond in the same language as the question."""


def answer_question(question: str, context: str = "") -> dict:
    """
    Answer a simple educational question using the local LLM.

    This is for EASY questions the user asks directly, without needing
    the full RAG + AI-Server pipeline. If context is provided (e.g. a
    course excerpt the user has open), it is injected into the prompt.

    Args:
        question: The student's question.
        context:  Optional text excerpt to ground the answer (e.g. a course page).

    Returns:
        dict with 'answer' (str) and 'has_context' (bool).
    """
    if context.strip():
        prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
    else:
        prompt = f"""Answer this educational question as clearly as possible.

Question: {question}

Answer:"""

    raw = model_manager.generate(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.4,
        max_tokens=512,
    )

    return {
        "answer": raw.strip(),
        "has_context": bool(context.strip()),
    }