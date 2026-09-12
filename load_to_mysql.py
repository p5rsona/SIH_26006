"""
Loads schema.sql + all CSVs in seed_csv/ into a real MySQL instance.

Usage:
    pip install mysql-connector-python --break-system-packages
    python3 load_to_mysql.py --host localhost --user root --password YOURPASS

Adjust connection args below or pass as CLI flags.
"""
import argparse
import csv
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
    ap.add_argument("--password", default="homeline786@") 
    ap.add_argument("--port", default=3306, type=int)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))

    conn = mysql.connector.connect(
        host=args.host, user=args.user, password=args.password, port=args.port
    )
    cur = conn.cursor()

    # Automatically wipe and recreate database to avoid table conflicts
    cur.execute("DROP DATABASE IF EXISTS sih_shipping")
    cur.execute("CREATE DATABASE sih_shipping")
    cur.execute("USE sih_shipping")

    print("Creating schema...")
    with open(os.path.join(here, "schema.sql")) as f:
        for statement in f.read().split(";"):
            statement = statement.strip()
            if statement:
                cur.execute(statement)
    conn.commit()

    for table in TABLE_ORDER:
        path = os.path.join(here, "seed_csv", f"{table}.csv")
        with open(path) as f:
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