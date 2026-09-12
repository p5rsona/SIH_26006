"""
check_db.py — is Supabase holding what we think it is?

    python check_db.py

Prints row counts per table and a breakdown of the freight rate series with
their provenance, so you can confirm in ten seconds that the database is
current. Worth running before any demo: every module reads through db.py and
falls back to CSV silently, so a stale or empty table looks like working
software right up until someone asks where a number came from.

Exit code is 1 if anything looks wrong, so it also works in a script.
"""

from __future__ import annotations

import sys

from config import REAL_INDEX_NAMES
from db import read_sql, status

# table -> rows expected after `python generate_data.py && python load_to_db.py`
EXPECTED = {
    "ports": 9,
    "origin_ports": 6,
    "routes": 54,
    "vessels": 30,
    "freight_rates_history": 4895,
    "commodity_prices": 1899,
    "fixtures": 250,
}


def main() -> int:
    print(status(), "\n")

    problems = []

    print(f"{'table':<24}{'rows':>7}{'expected':>10}")
    print("-" * 41)
    for table, expected in EXPECTED.items():
        df = read_sql(f"SELECT COUNT(1) AS n FROM {table}")
        if df is None:
            print(f"{table:<24}{'ERROR':>7}{expected:>10}")
            problems.append(f"{table}: query failed — is the schema loaded?")
            continue

        count = int(df.iloc[0]["n"])
        flag = "" if count == expected else "  <-- unexpected"
        print(f"{table:<24}{count:>7}{expected:>10}{flag}")
        if count != expected:
            problems.append(f"{table}: {count} rows, expected {expected}")

    series = read_sql(
        "SELECT index_name, COUNT(1) AS n, MIN(rate_date) AS first, MAX(rate_date) AS last "
        "FROM freight_rates_history GROUP BY index_name ORDER BY index_name"
    )
    if series is None or series.empty:
        problems.append("freight_rates_history: no series found")
    else:
        print(f"\n{'series':<20}{'rows':>6}  {'from':<12}{'to':<12}provenance")
        print("-" * 62)
        for _, row in series.iterrows():
            kind = "real (USDA)" if row["index_name"] in REAL_INDEX_NAMES else "generated"
            print(
                f"{row['index_name']:<20}{int(row['n']):>6}  "
                f"{str(row['first']):<12}{str(row['last']):<12}{kind}"
            )

        missing = REAL_INDEX_NAMES - set(series["index_name"])
        if missing:
            problems.append(
                f"real series missing from the database: {', '.join(sorted(missing))} "
                "— run fetch_real_data.py, then load_to_db.py"
            )

    # A zero rate is USDA's "no movement reported this week" placeholder, not a
    # price. data_loader filters it, but it should not be sitting in the table.
    zeros = read_sql("SELECT COUNT(1) AS n FROM freight_rates_history WHERE index_value <= 0")
    if zeros is not None and int(zeros.iloc[0]["n"]):
        print(f"\nnote: {int(zeros.iloc[0]['n'])} non-positive rate values "
              "(USDA no-movement placeholders — data_loader treats these as missing)")

    if problems:
        print("\nPROBLEMS")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nAll tables present and current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
