"""
HTTP routes for the tutoring conversation.

Unlike exam/slides, there is no Celery job here: each turn is fast
(one or two LLM calls), so we just call the service directly and
return the tutor's message right away. /start opens a new session
(and runs RAG once); /reply continues an existing session identified
by the thread_id returned from /start.
"""

from fastapi import APIRouter

from backend.ai_server.schemas.tutoring import TutoringStartRequest, TutoringReplyRequest
from backend.ai_server.services.tutoring_service import start_tutoring, reply_tutoring

router = APIRouter(prefix="/tutoring", tags=["tutoring"])


@router.post("/start")
def tutoring_start(payload: TutoringStartRequest):
    """Start a new tutoring session for the student's initial question."""
    return start_tutoring(payload)


@router.post("/reply")
def tutoring_reply(payload: TutoringReplyRequest):
    """Continue an existing tutoring session with the student's answer."""
    return reply_tutoring(payload)
