"""
Obsolete. The project has moved from local MySQL to Supabase (Postgres).

Use `load_to_db.py` instead — it creates the schema from schema_postgres.sql
and loads the same seed CSVs into the Supabase database configured in .env:

    pip install sqlalchemy psycopg2-binary
    python load_to_db.py

This file is kept only so old notes and scripts that mention it point
somewhere useful.
"""

import sys

MESSAGE = __doc__.strip()

if __name__ == "__main__":
    print(MESSAGE)
    sys.exit(1)
