"""
fetch_real_data.py — replaces synthetic series with real public data.

Two free sources, no API key:

1. World Bank "Pink Sheet" (CMO-Historical-Data-Monthly.xlsx)
   Monthly FOB benchmark prices back to 1960. We take Australian coal,
   US wheat (HRW) and maize — the three commodities in this project.
   -> commodity_prices_real.csv  (price_date, commodity, origin_id, price_usd_per_tonne)

2. USDA AMS "Grain Transportation Cost Indicators" (agtransport.usda.gov)
   Weekly transport cost indices back to 2002, including ocean vessel rates
   out of the US Gulf and the Pacific Northwest.
   -> freight_rates_real.csv     (rate_date, index_name, index_value)

Usage:
    python fetch_real_data.py --dry-run     # fetch + print what was found, write nothing
    python fetch_real_data.py               # write both CSVs
    python fetch_real_data.py --freight     # just the freight series
    python fetch_real_data.py --commodities # just the prices

Once the files exist, `person3_commodity_forecast.py` and `data_loader.py`
prefer them over the synthetic CSVs automatically.

NOT covered by these sources (still synthetic — say so to judges):
  * Baltic BDI/BCI/BPI/BSI indices — subscription only. Trading Economics has a
    limited free tier; that's the planned replacement.
  * Coal freight rates — USDA covers grain routes only.
  * Vessel fleet, port specs, fixtures — no free source; port specs are real but
    hand-collected.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

WORLD_BANK_XLSX = (
    "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/"
    "related/CMO-Historical-Data-Monthly.xlsx"
)
USDA_COST_INDICATORS = "https://agtransport.usda.gov/resource/8uye-ieij.json"

COMMODITY_PRICES_REAL = ROOT / "commodity_prices_real.csv"
FREIGHT_RATES_REAL = ROOT / "freight_rates_real.csv"

# Pink Sheet column -> (our commodity name, origin_ports.origin_id)
# Matching is fuzzy (lower-cased substring), because the header wording changes
# between releases, e.g. "Coal, Australian **" vs "Coal, Australia".
PINK_SHEET_COLUMNS = {
    "coal, austral": ("coal_newcastle", 1),
    "wheat, us hrw": ("wheat_gulf", 3),
    "maize": ("corn_gulf", 3),
}

# USDA field -> index_name stored in freight_rates_history
USDA_FIELDS = {
    "gulf_vessel": "USDA_GULF_VESSEL",
    "pacific_vessel": "USDA_PNW_VESSEL",
    "barge": "USDA_BARGE",
}


# =====================================================================
# 1. World Bank Pink Sheet — monthly commodity prices
# =====================================================================

def fetch_worldbank_prices(start_year: int = 2005) -> pd.DataFrame:
    """Monthly FOB prices, long format: price_date, commodity, origin_id,
    price_usd_per_tonne."""
    print(f"Downloading Pink Sheet ...\n  {WORLD_BANK_XLSX}")
    raw = pd.read_excel(WORLD_BANK_XLSX, sheet_name="Monthly Prices", header=None)

    # The sheet has a few title rows, then a row of commodity names, then units,
    # then data rows keyed like "1960M01". Find the first data row and treat the
    # rows above it as the header block.
    def is_period(value) -> bool:
        text = str(value).strip()
        return len(text) == 7 and text[4] in ("M", "m") and text[:4].isdigit()

    first_data_row = next((i for i, v in enumerate(raw[0]) if is_period(v)), None)
    if first_data_row is None:
        raise RuntimeError(
            "Could not find the 'YYYYMnn' date column in the Pink Sheet — "
            "the layout changed. Open the file and adjust fetch_worldbank_prices()."
        )

    header = (
        raw.iloc[:first_data_row]
        .fillna("")
        .astype(str)
        .apply(lambda col: " ".join(part for part in col if part and part != "nan"), axis=0)
    )
    data = raw.iloc[first_data_row:].copy()
    data.columns = list(header)

    date_col = data.columns[0]
    data["price_date"] = pd.to_datetime(
        data[date_col].astype(str).str.replace("M", "-", case=False), format="%Y-%m", errors="coerce"
    )
    data = data.dropna(subset=["price_date"])
    data = data[data["price_date"].dt.year >= start_year]

    rows = []
    matched = {}
    for pattern, (commodity, origin_id) in PINK_SHEET_COLUMNS.items():
        column = next((c for c in data.columns if pattern in str(c).lower()), None)
        if column is None:
            print(f"  !! no column matching '{pattern}' — skipped")
            continue
        matched[commodity] = column
        series = pd.to_numeric(data[column], errors="coerce")
        block = pd.DataFrame({
            "price_date": data["price_date"],
            "commodity": commodity,
            "origin_id": origin_id,
            "price_usd_per_tonne": series.round(2),
        }).dropna(subset=["price_usd_per_tonne"])
        rows.append(block)

    if not rows:
        raise RuntimeError("No commodity columns matched — check PINK_SHEET_COLUMNS.")

    print("  matched columns:")
    for commodity, column in matched.items():
        print(f"    {commodity:<16} <- {str(column).strip()[:70]}")

    out = pd.concat(rows).sort_values(["commodity", "price_date"]).reset_index(drop=True)
    out["price_date"] = out["price_date"].dt.strftime("%Y-%m-%d")
    return out


# =====================================================================
# 2. USDA — weekly ocean freight cost indices
# =====================================================================

def fetch_usda_freight(start_year: int = 2005, limit: int = 5000) -> pd.DataFrame:
    """Weekly transport cost indices, long format: rate_date, index_name,
    index_value."""
    url = f"{USDA_COST_INDICATORS}?$limit={limit}&$order=date%20ASC"
    print(f"Downloading USDA grain transportation cost indicators ...\n  {url}")
    with urllib.request.urlopen(url, timeout=60) as response:
        records = json.load(response)

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("USDA returned no rows.")
    df["rate_date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[df["rate_date"].dt.year >= start_year]

    blocks = []
    for field, index_name in USDA_FIELDS.items():
        if field not in df.columns:
            print(f"  !! field '{field}' missing from the USDA response — skipped")
            continue
        values = pd.to_numeric(df[field], errors="coerce")
        blocks.append(
            pd.DataFrame({
                "rate_date": df["rate_date"],
                "index_name": index_name,
                "index_value": values.round(2),
            }).dropna(subset=["index_value"])
        )

    out = pd.concat(blocks).sort_values(["index_name", "rate_date"]).reset_index(drop=True)
    out["rate_date"] = out["rate_date"].dt.strftime("%Y-%m-%d")
    return out


# =====================================================================
# CLI
# =====================================================================

def _summarise(df: pd.DataFrame, key: str, date_col: str, value_col: str) -> None:
    for name, group in df.groupby(key):
        print(
            f"    {name:<20} {len(group):>5} rows  "
            f"{group[date_col].min()} -> {group[date_col].max()}  "
            f"latest {float(group[value_col].iloc[-1]):,.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--commodities", action="store_true", help="fetch Pink Sheet prices only")
    parser.add_argument("--freight", action="store_true", help="fetch USDA freight indices only")
    parser.add_argument("--dry-run", action="store_true", help="fetch and report, write nothing")
    parser.add_argument("--start-year", type=int, default=2005)
    args = parser.parse_args()

    do_commodities = args.commodities or not args.freight
    do_freight = args.freight or not args.commodities
    failures = []

    if do_commodities:
        try:
            prices = fetch_worldbank_prices(args.start_year)
            print(f"  {len(prices)} monthly price rows")
            _summarise(prices, "commodity", "price_date", "price_usd_per_tonne")
            if not args.dry_run:
                prices.to_csv(COMMODITY_PRICES_REAL, index=False)
                print(f"  wrote {COMMODITY_PRICES_REAL.name}")
        except Exception as e:
            failures.append(f"commodity prices: {e}")
            print(f"  FAILED: {e}")

    if do_freight:
        try:
            freight = fetch_usda_freight(args.start_year)
            print(f"  {len(freight)} weekly freight rows")
            _summarise(freight, "index_name", "rate_date", "index_value")
            if not args.dry_run:
                freight.to_csv(FREIGHT_RATES_REAL, index=False)
                print(f"  wrote {FREIGHT_RATES_REAL.name}")
        except Exception as e:
            failures.append(f"freight indices: {e}")
            print(f"  FAILED: {e}")

    if failures:
        print("\nSome sources failed; the synthetic CSVs are still in place:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nDone. The forecasting modules now prefer the real files automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
