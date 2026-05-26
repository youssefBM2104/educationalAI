"""
wipe_db.py — Truncate all dev tables and reset sequences.
Run from project root: python3 scripts/wipe_db.py
"""

import sys
import os

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.config import settings
from sqlalchemy import create_engine, text

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

print("\nDone. All tables are empty, sequences reset to 1.")