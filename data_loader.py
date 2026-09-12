"""
data_loader.py — loads freight rate history for a route.

Tries Supabase first (see db.py), then the CSV seed
files (the same data). Routes are mapped to a Baltic index via
config.ROUTE_TO_INDEX, and the daily index is averaged to weekly. If neither
source has data for the route, falls back to a synthetic series so Person 2 can
build/test without being blocked on Person 1 (per the team's coordination
plan: "everyone else builds against mocked/static data in parallel").
"""

from __future__ import annotations

import os
import zlib

import numpy as np
import pandas as pd


from config import (
    FREIGHT_RATES_CSV,
    FREIGHT_RATES_REAL_CSV,
    FREIGHT_RATES_TABLE,
    RANDOM_SEED,
    ROUTE_TO_INDEX,
)


def _to_weekly(df: pd.DataFrame) -> pd.DataFrame | None:
    """Person 1's data is daily (business days); the models are weekly
    (W-MON). Average each week into one row labelled by its Monday."""
    if df is None or df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    weekly = (
        df.set_index("date")["rate"]
        .resample("W-MON")
        .mean()
        .interpolate()
        .dropna()
        .reset_index()
    )
    return weekly if len(weekly) else None


def _try_load_from_db(route: str) -> pd.DataFrame | None:
    """Loads the index series from the Supabase database (see db.py).
    Returns None if there's no database or no rows."""
    index_name = ROUTE_TO_INDEX.get(route)
    if index_name is None:
        return None

    from db import read_sql

    # column names per schema_postgres.sql
    df = read_sql(
        f"SELECT rate_date, index_value FROM {FREIGHT_RATES_TABLE} "
        "WHERE index_name = :index_name ORDER BY rate_date ASC",
        {"index_name": index_name},
    )
    if df is None or df.empty:
        return None
    return _to_weekly(df.rename(columns={"rate_date": "date", "index_value": "rate"}))


def _try_load_from_csv(route: str) -> pd.DataFrame | None:
    """Reads the index series from a CSV: the real file written by
    fetch_real_data.py first, then Person 1's synthetic seed file."""
    index_name = ROUTE_TO_INDEX.get(route)
    if index_name is None:
        return None

    for path in (FREIGHT_RATES_REAL_CSV, FREIGHT_RATES_CSV):
        if not os.path.exists(path):
            continue
        try:
            raw = pd.read_csv(path)
            raw = raw[raw["index_name"] == index_name]
            if raw.empty:
                continue
            weekly = _to_weekly(
                raw.rename(columns={"rate_date": "date", "index_value": "rate"})[["date", "rate"]]
            )
            if weekly is not None:
                weekly.attrs["file"] = os.path.basename(path)
                return weekly
        except Exception:
            continue
    return None


def generate_synthetic_series(
    route: str,
    n_weeks: int = 260,  # 5 years of weekly data
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Builds a synthetic-but-plausible weekly freight rate series with:
    - a slow trend
    - annual seasonality (monsoon-linked dip/spike)
    - autocorrelated noise (so it isn't pure white noise)
    - a correlated bunker (fuel) price series

    Route name is hashed into the seed so different routes get different
    but reproducible series. (zlib.crc32, not hash(): Python randomizes
    str hashes per process, so hash() gave a different series every run.)
    """
    rng = np.random.default_rng(seed + zlib.crc32(route.encode()) % 10_000)

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_weeks, freq="W-MON")
    n_weeks = len(dates)  # date_range can return periods-1 depending on end-date alignment
    t = np.arange(n_weeks)

    base_level = rng.uniform(8_000, 25_000)          # $/day, typical dry bulk range
    trend = base_level + t * rng.uniform(-5, 15)
    seasonal = 1500 * np.sin(2 * np.pi * t / 52 + rng.uniform(0, np.pi))

    # AR(1) noise so shocks persist a bit, like real freight markets
    noise = np.zeros(n_weeks)
    phi = 0.6
    sigma = base_level * 0.04
    for i in range(1, n_weeks):
        noise[i] = phi * noise[i - 1] + rng.normal(0, sigma)

    rate = np.clip(trend + seasonal + noise, 1_000, None)

    # Bunker price: correlated with rate level but its own slower dynamics
    bunker = 400 + 0.02 * trend + rng.normal(0, 15, n_weeks).cumsum() * 0.05
    bunker = np.clip(bunker, 250, 900)

    return pd.DataFrame({"date": dates, "rate": rate, "bunker_price": bunker})


def load_freight_rate_history(route: str) -> pd.DataFrame:
    """
    Main entry point. Returns a weekly DataFrame[date, rate, (bunker_price)]
    for a route. Source order: database -> real/synthetic CSV -> synthetic.
    The source used is recorded in df.attrs["source"].
    """
    for source, loader in (("database", _try_load_from_db), ("csv", _try_load_from_csv)):
        df = loader(route)
        if df is not None:
            df.attrs["source"] = df.attrs.get("file", source)
            return df

    df = generate_synthetic_series(route)
    df.attrs["source"] = "synthetic"
    return df


if __name__ == "__main__":
    for r in ["C5", "P1A_82", "TD3C"]:
        d = load_freight_rate_history(r)
        print(f"{r} [{d.attrs['source']}]: {len(d)} rows, {d['date'].min().date()} -> {d['date'].max().date()}")
        print(d.tail(3), "\n")
