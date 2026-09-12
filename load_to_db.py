"""
load_to_db.py — creates the schema and loads the seed CSVs into Supabase.

    # put the connection URL in .env first (copy .env.example), then:
    python load_to_db.py

Needs: pip install sqlalchemy psycopg2-binary

It drops and recreates the tables every run, then loads the CSVs from the
project folder (or seed_csv/ if that folder exists).

Flags:
    --schema-only   create the tables, load nothing
    --data-only     load the CSVs into tables that already exist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

# parents before children, so foreign keys resolve
TABLE_ORDER = [
    "ports", "origin_ports", "routes", "vessels",
    "freight_rates_history", "commodity_prices", "fixtures",
]

# tables whose primary key is an identity column
PK_COLUMNS = {
    "ports": "port_id",
    "origin_ports": "origin_id",
    "routes": "route_id",
    "vessels": "vessel_id",
    "freight_rates_history": "rate_id",
    "commodity_prices": "price_id",
    "fixtures": "fixture_id",
}

SCHEMA_FILE = "schema_postgres.sql"

# Real published data written by fetch_real_data.py, merged into the same
# tables as the synthetic seed rows: {table: (real csv, unique key)}.
#
# USDA freight indices use their own index_name values, so they sit alongside
# the synthetic BDI/BCI/BPI/BSI rows without clashing. Pink Sheet commodity
# prices DO overlap the generated ones on (price_date, commodity) — the real
# row wins, since it is concatenated second and de-duplication keeps the last.
# Both keys match the UNIQUE constraints in schema_postgres.sql, so this also
# prevents the load failing on a constraint violation.
REAL_CSV = {
    "freight_rates_history": ("freight_rates_real.csv", ["rate_date", "index_name"]),
    "commodity_prices": ("commodity_prices_real.csv", ["price_date", "commodity"]),
}


def _csv_dir() -> Path:
    seed = ROOT / "seed_csv"
    return seed if seed.is_dir() else ROOT


def _split_statements(sql: str) -> list[str]:
    """Naive split on ';' — fine for this schema (no stored procedures)."""
    return [part.strip() for part in sql.split(";") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Load schema + seed CSVs into Supabase")
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()

    import db

    engine = db.get_engine()
    if engine is None:
        print(f"Cannot reach the database: {db.status()}")
        print("Check SUPABASE_DB_URL in .env (see .env.example), then retry.")
        return 1

    print(db.status())
    schema_file = ROOT / SCHEMA_FILE
    csv_dir = _csv_dir()

    from sqlalchemy import text

    if not args.data_only:
        print(f"Creating schema from {schema_file.name} ...")
        statements = _split_statements(schema_file.read_text(encoding="utf-8"))
        with engine.begin() as conn:
            # children first, so foreign keys don't block the drop
            for table in reversed(TABLE_ORDER):
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
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

        real_spec = REAL_CSV.get(table)
        if real_spec:
            real_name, unique_key = real_spec
            real_path = csv_dir / real_name
            if real_path.exists():
                real_df = pd.read_csv(real_path)
                combined = len(df) + len(real_df)
                # real second, keep="last" -> published data supersedes generated
                df = pd.concat([df, real_df], ignore_index=True)
                df = df.drop_duplicates(subset=unique_key, keep="last").reset_index(drop=True)
                print(
                    f"  + {real_name}: {len(real_df)} real rows"
                    + (f", {combined - len(df)} synthetic superseded" if combined != len(df) else "")
                )
            else:
                print(f"  !! {real_name} not found — synthetic data only. Run fetch_real_data.py")

        df.to_sql(table, engine, if_exists="append", index=False, chunksize=1000)
        print(f"  loaded {len(df):>5} rows -> {table}")

        # Postgres identity columns don't advance when ids are supplied
        # explicitly, so the next insert would collide. Move the sequence past
        # the highest id we just loaded.
        pk = PK_COLUMNS.get(table)
        if pk and pk in df.columns:
            with engine.begin() as conn:
                conn.execute(text(
                    f"SELECT setval(pg_get_serial_sequence('{table}', '{pk}'), "
                    f"COALESCE((SELECT MAX({pk}) FROM {table}), 1))"
                ))

    print("\nDone. The forecasting, optimizer and API modules will now read from Supabase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
