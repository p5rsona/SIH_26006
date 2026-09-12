"""
db.py — one place that knows how to reach the database.

The project runs on **Supabase (Postgres)**. There is no MySQL backend any
more; the only thing to configure is the connection URL, which goes in a
`.env` file next to this one (copy `.env.example`).

Get the URL from the Supabase dashboard: open the project and click
**Connect** in the top bar, then pick a connection method. Copy the URI
verbatim and replace [YOUR-PASSWORD] — do not hand-assemble the host.

    Transaction pooler (use this one):
    SUPABASE_DB_URL=postgresql://postgres.<ref>:<password>@aws-<N>-<region>.pooler.supabase.com:6543/postgres

The pooler is IPv4-reachable, so it works on networks the direct connection
does not. Two details bite if you type the host by hand: the username must
be `postgres.<project-ref>`, not `postgres`, and the `aws-<N>` cluster index
is assigned per project — it is NOT derivable from the region, and it is not
always `aws-0`. Any mismatch in host, index or username returns the same
unhelpful `FATAL: (ENOTFOUND) tenant/user ... not found`, which looks like a
credentials problem but is really "this pooler does not host that project".

    Direct connection (usually unusable):
    SUPABASE_DB_URL=postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres

db.<ref>.supabase.co resolves to IPv6 only. On an IPv4-only network it fails
with "network is unreachable" no matter how correct the credentials are.

The individual parts (PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_NAME) also
work if you'd rather not paste a URL.

Every reader in the project calls `read_sql()` and falls back to CSV when it
returns None, so the project keeps working with no database at all.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parent

# Why this query failed / why the engine is None. Surfaced by status().
last_error: str = ""


# Tried in order. `.env.txt` is there because Windows likes to append .txt
# when you save or rename a file in Explorer.
ENV_FILES = (".env", ".env.local", ".env.txt")


def env_file_path() -> Optional[Path]:
    """The settings file actually being used, or None if there isn't one."""
    for name in ENV_FILES:
        candidate = ROOT / name
        if candidate.is_file():
            return candidate
    return None


def load_env(path: Path | None = None) -> None:
    """Reads KEY=VALUE lines from the .env file into os.environ (existing
    environment variables win).

    Uses python-dotenv when it's installed, otherwise a small parser, so the
    project has no hard dependency on it.
    """
    env_file = path or env_file_path()
    if env_file is None or not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
        return
    except ImportError:
        pass

    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


load_env()


def backend() -> str:
    """Kept so older callers still work. Always 'postgres'."""
    return "postgres"


def database_url() -> str:
    """SQLAlchemy URL for the Supabase/Postgres database."""
    url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if url:
        # SQLAlchemy wants the psycopg2 driver spelled out
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg2://", 1)
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+psycopg2://", 1)
        return url

    return (
        "postgresql+psycopg2://"
        f"{os.getenv('PG_USER', 'postgres')}:{os.getenv('PG_PASSWORD', '')}"
        f"@{os.getenv('PG_HOST', 'localhost')}:{os.getenv('PG_PORT', '5432')}"
        f"/{os.getenv('PG_NAME', 'postgres')}"
    )


def _connect_args() -> dict:
    return {
        "connect_timeout": int(os.getenv("PG_CONNECT_TIMEOUT", "8")),
        "sslmode": os.getenv("PG_SSLMODE", "require"),
    }


@lru_cache(maxsize=1)
def get_engine():
    """A SQLAlchemy engine, or None if the driver is missing or Supabase
    can't be reached. Cached, including the failure, so we don't retry a dead
    connection on every call. The reason lands in `last_error`."""
    global last_error
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(database_url(), pool_pre_ping=True, connect_args=_connect_args())
        with engine.connect() as conn:          # fail fast if it isn't reachable
            conn.execute(text("SELECT 1"))
        last_error = ""
        return engine
    except ImportError as exc:
        last_error = f"missing driver ({exc}) — pip install sqlalchemy psycopg2-binary"
    except Exception as exc:
        last_error = _explain(exc)
    return None


def _explain(exc: Exception) -> str:
    """Turns the usual connection failures into something actionable."""
    text_ = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    host = os.getenv("PG_HOST", "") or database_url().split("@")[-1].split("/")[0]
    lowered = text_.lower()
    if "network is unreachable" in lowered or "cannot assign requested address" in lowered or "timeout expired" in lowered:
        if ".supabase.co" in host and "pooler" not in host:
            return (
                f"{text_} — db.*.supabase.co resolves to IPv6 only. On an IPv4-only "
                "network use the pooler URI instead: Supabase dashboard -> Connect "
                "-> Transaction pooler (port 6543)."
            )
    if "password authentication failed" in lowered:
        return f"{text_} — check the password in .env (reset it in Project Settings -> Database)."
    if "tenant" in lowered and "not found" in lowered:
        return (
            f"{text_} — this pooler does not host that project. Do not edit the host by "
            "hand: copy the URI from the dashboard -> Connect -> Transaction pooler. The "
            "aws-<N> index is per-project and not derivable from the region, and the "
            "username must be postgres.<project-ref>."
        )
    if "does not exist" in lowered and "relation" in lowered:
        return f"{text_} — the schema is not loaded yet. Run: python load_to_db.py"
    return text_


def read_sql(query: str, params: Optional[Sequence[Any]] = None) -> Optional[pd.DataFrame]:
    """Runs a query and returns a DataFrame, or None if there's no database.

    Use `:name` style parameters:
        read_sql("SELECT * FROM t WHERE index_name = :name", {"name": "BCI"})
    """
    global last_error
    engine = get_engine()
    if engine is None:
        return None
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params or {})
    except Exception as exc:
        last_error = _explain(exc)
        return None


def status() -> str:
    engine = get_engine()
    where = database_url().split("@")[-1]           # never print the password
    if engine is not None:
        return f"supabase @ {where}: connected"
    hint = last_error
    if not os.getenv("SUPABASE_DB_URL") and not os.getenv("DATABASE_URL"):
        found = env_file_path()
        hint = (
            f"SUPABASE_DB_URL is not set — {'read ' + found.name + ', but it has no SUPABASE_DB_URL line' if found else 'no ' + ENV_FILES[0] + ' file in ' + str(ROOT)}"
        )
    return f"supabase @ {where}: unreachable (using CSV fallback) — {hint}"


if __name__ == "__main__":
    print(status())
    df = read_sql("SELECT COUNT(*) AS n FROM vessels")
    if df is None:
        # None means the query failed, which is not the same as "no database".
        # Without this the connection looks broken when the schema is missing.
        print("vessels: unavailable —", last_error or "unknown error")
    else:
        print("vessels:", int(df.iloc[0]["n"]))
