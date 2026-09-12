"""
db.py — one place that knows how to reach the database.

Supports two backends, chosen with the DB_BACKEND environment variable:

    DB_BACKEND=mysql       (default)  -> local MySQL, as before
    DB_BACKEND=postgres               -> Postgres / Supabase

Supabase connection settings (Project Settings -> Database -> Connection string):

    set DB_BACKEND=postgres
    set SUPABASE_DB_URL=postgresql://postgres.<ref>:<password>@<host>:6543/postgres

or the individual parts: PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_NAME.

Every reader in the project calls `read_sql()` and falls back to CSV when it
returns None, so the project keeps working with no database at all.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional, Sequence

import pandas as pd

from config import DB_CONFIG


def backend() -> str:
    """'mysql' or 'postgres'. 'supabase' is accepted as an alias."""
    value = os.getenv("DB_BACKEND", "mysql").strip().lower()
    return "postgres" if value in {"postgres", "postgresql", "supabase"} else "mysql"


def database_url() -> str:
    """SQLAlchemy URL for the active backend."""
    if backend() == "postgres":
        url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
        if url:
            # SQLAlchemy wants the psycopg2 driver spelled out
            return url.replace("postgresql://", "postgresql+psycopg2://", 1) if url.startswith("postgresql://") else url
        return (
            "postgresql+psycopg2://"
            f"{os.getenv('PG_USER', 'postgres')}:{os.getenv('PG_PASSWORD', '')}"
            f"@{os.getenv('PG_HOST', 'localhost')}:{os.getenv('PG_PORT', '5432')}"
            f"/{os.getenv('PG_NAME', 'postgres')}"
        )

    return (
        "mysql+mysqlconnector://"
        f"{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )


@lru_cache(maxsize=1)
def get_engine():
    """A SQLAlchemy engine, or None if the driver is missing or the database
    can't be reached. Cached, including the failure, so we don't retry a dead
    connection on every call."""
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(database_url(), pool_pre_ping=True, connect_args=_connect_args())
        with engine.connect() as conn:          # fail fast if it isn't reachable
            conn.execute(text("SELECT 1"))
        return engine
    except Exception:
        return None


def _connect_args() -> dict:
    if backend() == "postgres":
        return {"connect_timeout": 5, "sslmode": os.getenv("PG_SSLMODE", "require")}
    return {"connection_timeout": 5}


def read_sql(query: str, params: Optional[Sequence[Any]] = None) -> Optional[pd.DataFrame]:
    """Runs a query and returns a DataFrame, or None if there's no database.

    Use `:name` style parameters so the same SQL works on both backends:
        read_sql("SELECT * FROM t WHERE index_name = :name", {"name": "BCI"})
    """
    engine = get_engine()
    if engine is None:
        return None
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params or {})
    except Exception:
        return None


def status() -> str:
    engine = get_engine()
    where = database_url().split("@")[-1]           # never print the password
    return f"{backend()} @ {where}: " + ("connected" if engine is not None else "unreachable (using CSV fallback)")


if __name__ == "__main__":
    print(status())
    df = read_sql("SELECT COUNT(*) AS n FROM vessels")
    print("vessels:", None if df is None else int(df.iloc[0]["n"]))
