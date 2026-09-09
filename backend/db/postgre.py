import enum

from sqlalchemy import DateTime, String, func, Enum, Integer, Float
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

class Image(Base):
    __tablename__ = "images"
    image_id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False)
    chunk_id = Column(String, nullable=True)
    page_number = Column(Integer)
    bbox_left = Column(Float)
    bbox_top = Column(Float)
    bbox_right = Column(Float)
    bbox_bottom = Column(Float)
    minio_path = Column(String, nullable=False)
    vlm_description = Column(String)
    created_at = Column(DateTime, server_default=func.now())

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
