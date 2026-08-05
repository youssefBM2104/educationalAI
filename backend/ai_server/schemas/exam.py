"""
Request schema for the exam-generation endpoint.

This is the "form" that whoever calls the AI-Server has to fill in.
It intentionally stays minimal: `query` is free text (e.g. "Generate 5 MCQ
questions about photosynthesis") because the existing `intake_agent`
already knows how to parse question type and count out of that text.
We don't re-implement that parsing here — we just pass the text through.
"""

from pydantic import BaseModel


class ExamRequest(BaseModel):
    user_id: str
    course_id: str
    query: str
