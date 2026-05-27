"""
wipe_db.py — Truncate all dev tables and reset sequences.
Run from project root: python3 scripts/wipe_db.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.config import settings
from sqlalchemy import create_engine, text
from neo4j import GraphDatabase


# --- Wipe PostgreSQL ---
TABLES = [
    "documents",
    # add future tables here as the schema grows
    # "users",
    # "courses",
]

engine = create_engine(settings.postgres_url)

with engine.begin() as conn:
    for table in TABLES:
        conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;"))
        print(f"✓  Wiped Postgres table: {table}")


# --- Wipe Neo4j ---
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=(settings.neo4j_user, settings.neo4j_password)
)

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    print("✓ Wiped Neo4j: all nodes and relationships deleted")

driver.close()

# ── 2. Qdrant ─────────────────────────────────────────────────────────────────

from qdrant_client import QdrantClient

qdrant = QdrantClient(settings.qdrant_url)
collection = settings.qdrant_collection

existing = [c.name for c in qdrant.get_collections().collections]
if collection in existing:
    qdrant.delete_collection(collection)
    print(f"✓  Deleted Qdrant collection: {collection}")
else:
    print(f"–  Qdrant collection not found, nothing to delete: {collection}")

# ── 3. MinIO ──────────────────────────────────────────────────────────────────

from minio import Minio
from minio.deleteobjects import DeleteObject

minio = Minio(
    settings.minio_endpoint,
    access_key=settings.minio_root_user,
    secret_key=settings.minio_root_password,
    secure=False,
)

for bucket in (settings.minio_bucket_originals, settings.minio_bucket_markdown):
    if not minio.bucket_exists(bucket):
        print(f"–  MinIO bucket not found, nothing to delete: {bucket}")
        continue

    objects = list(minio.list_objects(bucket, recursive=True))
    if objects:
        delete_list = [DeleteObject(o.object_name) for o in objects]
        errors = list(minio.remove_objects(bucket, iter(delete_list)))
        if errors:
            for err in errors:
                print(f"  ✗  Error deleting {err.name}: {err.message}")
        print(f"✓  Wiped MinIO bucket: {bucket} ({len(objects)} object(s) removed)")
    else:
        print(f"–  MinIO bucket already empty: {bucket}")

# ─────────────────────────────────────────────────────────────────────────────

print("\nDone. Postgres tables empty, Qdrant collection deleted, MinIO buckets empty, Neo4j is reset.")
print("The Qdrant collection and MinIO buckets will be recreated on next ingestion.")
