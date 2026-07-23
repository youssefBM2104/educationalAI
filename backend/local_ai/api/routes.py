from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

from backend.local_ai.model_manager import model_manager
from backend.local_ai.tasks.qa import answer_question
from backend.local_ai.tasks.conversation_summary import summarize_conversation
from backend.local_ai.tasks.learning_materials import get_mindmap, get_summary

router = APIRouter(prefix="/local-ai", tags=["Local AI"])

# Base URL for internal calls to the agents pipeline
INTERNAL_API_URL = "http://localhost:8000"


# ── Schemas ───────────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    model_ready:    bool
    ollama_running: bool
    detail:         str

class QARequest(BaseModel):
    question: str           = Field(..., example="What is Newton's second law?")
    context:  Optional[str] = Field("", example="Optional course excerpt to ground the answer")

class ConversationMessage(BaseModel):
    role:    Literal["user", "assistant"]
    content: str

class ConversationSummaryRequest(BaseModel):
    messages: List[ConversationMessage] = Field(..., min_length=1)

class MindmapRequest(BaseModel):
    query: str = Field(..., example="Explain the water cycle")

class LearningSummaryRequest(BaseModel):
    query:        str                               = Field(..., example="Photosynthesis")
    detail_level: Literal["short", "medium", "long"] = Field("medium")
    course_title: Optional[str]                     = Field("", example="Biology 101")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status", response_model=StatusResponse)
def get_status():
    """
    Check whether the local model is ready to handle requests.

    - If the fine-tuned adapter exists: model_ready=True, no Ollama needed.
    - If only Ollama fallback: model_ready depends on Ollama running.
    - Frontend should call this on startup to guide the user.
    """
    model_ready    = model_manager.check_model_ready()
    ollama_running = model_manager.check_ollama_running()

    if model_ready:
        detail = "Fine-tuned adapter loaded and ready."
    elif ollama_running:
        detail = "Using Ollama fallback (fine-tuned adapter not found)."
    else:
        detail = (
            "Model not ready. Either run 'python -m training.train' "
            "or install Ollama at https://ollama.com."
        )

    return StatusResponse(
        model_ready=model_ready,
        ollama_running=ollama_running,
        detail=detail,
    )


@router.post("/setup")
def setup_model():
    """
    Only needed when using the Ollama fallback (before training/train.py has been run).
    Checks that Ollama is running — the model pull is handled by Ollama itself.
    Once you have run training/train.py, this endpoint is no longer needed.
    """
    if model_manager.check_model_ready():
        return {"message": "Fine-tuned adapter is already loaded. No setup needed."}

    if not model_manager.check_ollama_running():
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is not running. "
                "Install it at https://ollama.com and start it, "
                "or run 'python -m training.train' to use the fine-tuned adapter."
            ),
        )
    return {
        "message": (
            "Ollama is running. The model will be pulled automatically on first use. "
            "Run 'python -m training.train' when ready to switch to the fine-tuned adapter."
        )
    }


@router.post("/qa")
def qa(request: QARequest):
    """
    Answer a simple educational question using the local small AI model.
    Runs entirely on the user's machine — no data sent to the server.

    Optionally accepts a 'context' excerpt to ground the answer
    (e.g. a course paragraph the user has open).

    Returns: { "answer": "...", "has_context": bool }
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_question(
            question=request.question,
            context=request.context or "",
        )
        return {"source": "local", "result": result}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/conversation-summary")
def conversation_summary(request: ConversationSummaryRequest):
    """
    Summarize a tutoring conversation locally using the small AI model.
    Runs entirely on the user's machine — no data sent to the server.
    Useful for session recaps and compressing history before sending to AI-Server.

    Returns: { "recap": "...", "topics_covered": [...], "open_questions": [...] }
    """
    messages = [m.model_dump() for m in request.messages]

    try:
        result = summarize_conversation(messages=messages)
        return {"source": "local", "result": result}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/mindmap")
async def mindmap(request: MindmapRequest):
    """
    Generate a mindmap for a given topic.
    Delegates to the existing Mindmap Agent pipeline in agents/:
      query → RAG (Qdrant + Neo4j) → Mindmap Agent → Export (Mermaid + HTML)

    Returns: { "mermaid": "...", "html": "..." }
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = await get_mindmap(
            query=request.query,
            internal_api_url=INTERNAL_API_URL,
        )
        return {"source": "agents", "result": result}
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/learning-summary")
async def learning_summary(request: LearningSummaryRequest):
    """
    Generate a structured summary for a given topic.
    Delegates to the existing Summary pipeline in agents/:
      query → RAG (Qdrant + Neo4j) → Extractor Agent → Writer Agent → Summary

    detail_level controls output length:
      short  → ~150 words
      medium → ~400 words
      long   → ~800 words

    Returns: { "summary": "...", "detail_level": "..." }
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = await get_summary(
            query=request.query,
            detail_level=request.detail_level,
            course_title=request.course_title or "",
            internal_api_url=INTERNAL_API_URL,
        )
        return {"source": "agents", "result": result}
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))