"""
Loads schema.sql + Person 1's seed CSVs into a real MySQL instance.
CSVs are read from seed_csv/ if that folder exists, otherwise from the
project folder itself (where they currently live).

Usage:
    pip install mysql-connector-python
    python load_to_mysql.py --host localhost --user root --password YOURPASS
    (or set FR_DB_PASSWORD instead of passing --password; if neither is
    given you'll be prompted)

Adjust connection args below or pass as CLI flags.
"""
import argparse
import csv
import getpass
import os

import mysql.connector

TABLE_ORDER = [
    "ports", "origin_ports", "routes", "vessels",
    "freight_rates_history", "commodity_prices", "fixtures",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--user", default="root")
    # never hardcode the password in the repo — env var or prompt instead
    ap.add_argument("--password", default=os.getenv("FR_DB_PASSWORD"))
    ap.add_argument("--port", default=3306, type=int)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(here, "seed_csv")
    if not os.path.isdir(csv_dir):
        csv_dir = here

    if args.password is None:
        args.password = getpass.getpass(f"MySQL password for {args.user}@{args.host}: ")

    conn = mysql.connector.connect(
        host=args.host, user=args.user, password=args.password, port=args.port
    )
    cur = conn.cursor()

    # Automatically wipe and recreate database to avoid table conflicts
    cur.execute("DROP DATABASE IF EXISTS sih_shipping")
    cur.execute("CREATE DATABASE sih_shipping")
    cur.execute("USE sih_shipping")

    print("Creating schema...")
    with open(os.path.join(here, "schema.sql"), encoding="utf-8") as f:
        for statement in f.read().split(";"):
            statement = statement.strip()
            if statement:
                cur.execute(statement)
    conn.commit()

    for table in TABLE_ORDER:
        path = os.path.join(csv_dir, f"{table}.csv")
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                continue
            cols = list(rows[0].keys())
            placeholders = ",".join(["%s"] * len(cols))
            sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
            values = [tuple((None if v == "" else v) for v in row.values()) for row in rows]
            cur.executemany(sql, values)
            conn.commit()
            print(f"  loaded {len(rows)} rows -> {table}")

    cur.close()
    conn.close()
    print("Done. Database 'sih_shipping' is ready.")

if __name__ == "__main__":
    main()