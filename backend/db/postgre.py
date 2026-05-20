import enum

from sqlalchemy import DateTime, String, func, Enum
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.core.config import settings

engine = create_engine(settings.postgres_url)

Base = declarative_base()

class DocumentStatus(str, enum.Enum):
    pending    = "pending"
    processing = "processing"
    ready      = "ready"
    failed     = "failed"


class Document(Base):
    __tablename__ = "documents"
    id = Column(String,primary_key=True)
    filename = Column(String,nullable=False)
    course_id = Column(String)
    status = Column(Enum(DocumentStatus))
    created_at = Column(DateTime,server_default=func.now())
    sha256_hash = Column(String)
    minio_key = Column(String)
    error_msg = Column(String)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
