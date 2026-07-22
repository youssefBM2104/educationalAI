"""
Request schemas for the tutoring endpoints.

Tutoring is not a one-shot job like exam/slides — it's a back-and-forth
conversation (Socratic scaffolding), so it needs two different entry
points instead of one:

- TutoringStartRequest: the very first turn. Only here do we run the RAG
  retrieval step, since the tutoring graph's checkpointer will remember
  the retrieved context for every later turn of the same session.
- TutoringReplyRequest: every following turn. It only carries the
  student's answer plus the `thread_id` handed back by /tutoring/start,
  which tells LangGraph's checkpointer which conversation to resume.
"""

from pydantic import BaseModel


class TutoringStartRequest(BaseModel):
    user_id: str
    course_id: str
    query: str


class TutoringReplyRequest(BaseModel):
    thread_id: str
    student_answer: str
