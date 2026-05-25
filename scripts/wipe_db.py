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
        print(f"✓ Wiped table: {table}")


# --- Wipe Neo4j ---
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=(settings.neo4j_user, settings.neo4j_password)
)

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    print("✓ Wiped Neo4j: all nodes and relationships deleted")

driver.close()

print("\nDone. All tables are empty, sequences reset to 1.")