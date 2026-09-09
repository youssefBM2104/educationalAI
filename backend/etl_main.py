from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db.postgre import init_db
from backend.api.routes_documents import router as documents_router
from backend.api.routes_shared import router as shared_router

app = FastAPI(title="ETL Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(documents_router)
app.include_router(shared_router)


@app.get("/health")
def health():
    return {"status": "ok"}
