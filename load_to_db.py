"""
load_to_db.py — creates the schema and loads the seed CSVs into whichever
database db.py is pointed at: local MySQL, or Postgres/Supabase.

    # local MySQL (same as load_to_mysql.py)
    set DB_BACKEND=mysql
    set FR_DB_PASSWORD=yourpassword
    python load_to_db.py

    # Supabase (Project Settings -> Database -> Connection string -> URI)
    set DB_BACKEND=postgres
    set SUPABASE_DB_URL=postgresql://postgres.<ref>:<password>@<host>:6543/postgres
    python load_to_db.py

Needs: pip install sqlalchemy psycopg2-binary   (Postgres)
       pip install sqlalchemy mysql-connector-python   (MySQL)

It drops and recreates the tables every run, then loads the CSVs from the
project folder (or seed_csv/ if that folder exists).

MySQL note: the `sih_shipping` database must already exist (CREATE DATABASE
sih_shipping;) because the connection selects it. `load_to_mysql.py` still
works too and creates the database for you.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

# parents before children, so foreign keys resolve
TABLE_ORDER = [
    "ports", "origin_ports", "routes", "vessels",
    "freight_rates_history", "commodity_prices", "fixtures",
]

# tables whose primary key is an identity/auto-increment column
PK_COLUMNS = {
    "ports": "port_id",
    "origin_ports": "origin_id",
    "routes": "route_id",
    "vessels": "vessel_id",
    "freight_rates_history": "rate_id",
    "commodity_prices": "price_id",
    "fixtures": "fixture_id",
}


def _csv_dir() -> Path:
    seed = ROOT / "seed_csv"
    return seed if seed.is_dir() else ROOT


def _split_statements(sql: str) -> list[str]:
    """Naive split on ';' — fine for this schema (no stored procedures)."""
    return [part.strip() for part in sql.split(";") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Load schema + seed CSVs into MySQL or Supabase")
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()

    import db

    engine = db.get_engine()
    if engine is None:
        print(f"Cannot reach the database: {db.status()}")
        print("Check DB_BACKEND and the connection settings (see db.py), then retry.")
        return 1

    print(db.status())
    is_postgres = db.backend() == "postgres"
    schema_file = ROOT / ("schema_postgres.sql" if is_postgres else "schema.sql")
    csv_dir = _csv_dir()

    from sqlalchemy import text

    if not args.data_only:
        print(f"Creating schema from {schema_file.name} ...")
        statements = _split_statements(schema_file.read_text(encoding="utf-8"))
        cascade = " CASCADE" if is_postgres else ""
        with engine.begin() as conn:
            # children first, so foreign keys don't block the drop
            for table in reversed(TABLE_ORDER):
                conn.execute(text(f"DROP TABLE IF EXISTS {table}{cascade}"))
            for statement in statements:
                conn.execute(text(statement))
        print(f"  {len(statements)} statements executed")

    if args.schema_only:
        return 0

    for table in TABLE_ORDER:
        path = csv_dir / f"{table}.csv"
        if not path.exists():
            print(f"  !! {path.name} not found — skipped")
            continue
        df = pd.read_csv(path)
        df.to_sql(table, engine, if_exists="append", index=False, chunksize=1000)
        print(f"  loaded {len(df):>5} rows -> {table}")

        # Postgres identity columns don't advance when ids are supplied
        # explicitly, so the next insert would collide. Move the sequence past
        # the highest id we just loaded.
        pk = PK_COLUMNS.get(table)
        if is_postgres and pk and pk in df.columns:
            with engine.begin() as conn:
                conn.execute(text(
                    f"SELECT setval(pg_get_serial_sequence('{table}', '{pk}'), "
                    f"COALESCE((SELECT MAX({pk}) FROM {table}), 1))"
                ))

    print("\nDone. The forecasting, optimizer and API modules will now read from the database.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
