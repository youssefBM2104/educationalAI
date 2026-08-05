"""
HTTP routes for exam generation.

This file only knows about FastAPI and JSON. It validates the incoming
request (via ExamRequest), hands the actual work off to Celery, and
returns a job_id right away instead of blocking the caller for however
long the multi-agent pipeline takes. Progress/result is fetched via the
/jobs/{job_id} polling endpoint.
"""

from fastapi import APIRouter
from celery.result import AsyncResult

from backend.ai_server.schemas.exam import ExamRequest
from backend.ai_server.ai_server_tasks import celery_app, generate_exam_task

router = APIRouter(prefix="/exam", tags=["exam"])


@router.post("/generate", status_code=202)
def create_exam(payload: ExamRequest):
    """Queue an exam-generation job. Returns a job_id to poll."""
    task = generate_exam_task.delay(payload.model_dump())
    return {"job_id": task.id}


@router.get("/jobs/{job_id}")
def get_exam_job(job_id: str):
    """Check the status (and result, once ready) of a queued exam job."""
    result = AsyncResult(job_id, app=celery_app)

    if result.state == "SUCCESS":
        return {"status": "done", "exam_set": result.result}
    if result.state == "FAILURE":
        return {"status": "failed", "error": str(result.result)}
    return {"status": result.state.lower()}
