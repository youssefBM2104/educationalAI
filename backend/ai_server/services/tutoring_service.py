"""
Business logic for the tutoring conversation.

Unlike exam/slides, tutoring is not a single job — it's a multi-turn
Socratic dialogue. get_tutoring_graph() is compiled with a MemorySaver
checkpointer, so LangGraph remembers where a given conversation stands
as long as we keep passing the same `thread_id` back in.

- start_tutoring: first turn only. Runs RAG retrieval once (rag_chunks
  and kg_context get stored in the checkpointer's state for this
  thread_id, so later turns don't need to repeat it), then invokes the
  graph with the student's initial question.
- reply_tutoring: every following turn. Only the student's answer is
  sent in; the checkpointer already has the rest of the state.

Known limitation: MemorySaver keeps state in this process's RAM. That's
fine for a single AI-Server process (current setup), but if this ever
runs behind multiple worker processes, a later turn could land on a
different process and lose the conversation. At that point, swap
MemorySaver for a persistent LangGraph checkpointer (Redis/Postgres).
"""

import logging
import uuid

from backend.agents.retrieve import retrieve_node
from backend.agents.tutoring.tutoring_graph import get_tutoring_graph
from backend.ai_server.schemas.tutoring import TutoringStartRequest, TutoringReplyRequest

logger = logging.getLogger(__name__)


def start_tutoring(request: TutoringStartRequest) -> dict:
    """Start a new tutoring session and return its thread_id + first message."""
    thread_id = str(uuid.uuid4())
    state = {
        "user_id": request.user_id,
        "course_id": request.course_id,
        "query": request.query,
    }

    logger.info("Tutoring start: retrieving context for course_id=%s", request.course_id)
    state.update(retrieve_node(state))

    config = {"configurable": {"thread_id": thread_id}}
    result = get_tutoring_graph().invoke(state, config)

    logger.info("Tutoring start: session %s created", thread_id)
    return {"thread_id": thread_id, "message": result["tutor_message"]}


def reply_tutoring(request: TutoringReplyRequest) -> dict:
    """Continue an existing tutoring session with the student's answer."""
    config = {"configurable": {"thread_id": request.thread_id}}
    result = get_tutoring_graph().invoke(
        {"student_answer": request.student_answer}, config
    )

    logger.info("Tutoring reply: session %s advanced", request.thread_id)
    return {"thread_id": request.thread_id, "message": result["tutor_message"]}
