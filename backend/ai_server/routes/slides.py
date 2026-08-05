"""
HTTP routes for slide-deck (lecture) generation.

Same job/polling pattern as routes/exam.py: validate the request, queue
it on Celery, return a job_id immediately. The slide-generation pipeline
runs planner -> content_generator -> slide_builder -> export, which can
take a while, so we never run it inline in the request handler.
"""

from fastapi import APIRouter
from celery.result import AsyncResult

from backend.ai_server.schemas.slides import SlidesRequest
from backend.ai_server.ai_server_tasks import celery_app, generate_slides_task

router = APIRouter(prefix="/slides", tags=["slides"])


@router.post("/generate", status_code=202)
def create_slides(payload: SlidesRequest):
    """Queue a slide-generation job. Returns a job_id to poll."""
    task = generate_slides_task.delay(payload.model_dump())
    return {"job_id": task.id}


@router.get("/jobs/{job_id}")
def get_slides_job(job_id: str):
    """Check the status (and result, once ready) of a queued slides job."""
    result = AsyncResult(job_id, app=celery_app)

    if result.state == "SUCCESS":
        return {"status": "done", **result.result}
    if result.state == "FAILURE":
        return {"status": "failed", "error": str(result.result)}
    return {"status": result.state.lower()}
