from backend.local_ai.model_manager import model_manager
from typing import List

SYSTEM_PROMPT = """You are an educational assistant.
You summarize conversations between a student and an AI tutor.
Your goal is to extract what was learned, what questions were asked,
and what topics still need clarification.
Always respond in the same language as the conversation."""


def summarize_conversation(messages: List[dict]) -> dict:
    """
    Summarize a conversation history between a student and the AI tutor.

    Runs entirely locally — no data is sent to the server.
    Useful to give the user a recap of their session, or to compress
    history before sending it to the AI-Server.

    Args:
        messages: List of dicts with keys 'role' ('user' | 'assistant')
                  and 'content' (str).
                  Example:
                  [
                    {"role": "user",      "content": "What is photosynthesis?"},
                    {"role": "assistant", "content": "Photosynthesis is ..."},
                  ]

    Returns:
        dict with:
          - 'recap'           : short paragraph of what was discussed
          - 'topics_covered'  : list of topics that came up
          - 'open_questions'  : list of questions that were not fully resolved
    """
    if not messages:
        return {
            "recap": "No conversation to summarize.",
            "topics_covered": [],
            "open_questions": [],
        }

    # Format the conversation as a readable transcript
    transcript_lines = []
    for msg in messages:
        role = "Student" if msg.get("role") == "user" else "Tutor"
        transcript_lines.append(f"{role}: {msg.get('content', '').strip()}")
    transcript = "\n".join(transcript_lines)

    prompt = f"""Here is a tutoring conversation transcript:

{transcript}

Reply EXACTLY in this format:

RECAP:
(1-2 sentences summarizing what was discussed overall)

TOPICS COVERED:
- topic 1
- topic 2

OPEN QUESTIONS:
- question that was not fully resolved (or "None" if all were answered)"""

    raw = model_manager.generate(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.3,
        max_tokens=512,
    )

    # Parse sections
    recap = ""
    topics_covered: List[str] = []
    open_questions: List[str] = []

    def extract_section(text: str, header: str, next_header: str) -> str:
        if header not in text:
            return ""
        part = text.split(header)[1]
        if next_header and next_header in part:
            part = part.split(next_header)[0]
        return part.strip()

    def parse_bullets(block: str) -> List[str]:
        lines = block.strip().split("\n")
        return [l.lstrip("- ").strip() for l in lines if l.strip() and l.strip() != "None"]

    recap         = extract_section(raw, "RECAP:", "TOPICS COVERED:")
    topics_block  = extract_section(raw, "TOPICS COVERED:", "OPEN QUESTIONS:")
    open_block    = extract_section(raw, "OPEN QUESTIONS:", "")

    topics_covered = parse_bullets(topics_block)
    open_questions = parse_bullets(open_block)

    return {
        "recap": recap,
        "topics_covered": topics_covered,
        "open_questions": open_questions,
        "raw": raw,
    }