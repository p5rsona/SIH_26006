"""
data_loader.py — loads freight rate history for a route.

Tries Person 1's MySQL DB first. If it's unreachable or the route has no
rows yet, falls back to a synthetic-but-realistic series so Person 2 can
build/test without being blocked on Person 1 (per the team's coordination
plan: "everyone else builds against mocked/static data in parallel").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DB_CONFIG, FREIGHT_RATES_TABLE, RANDOM_SEED


def _try_load_from_mysql(route: str) -> pd.DataFrame | None:
    """Attempt to load real data from MySQL. Returns None on any failure."""
    try:
        import mysql.connector  # optional dependency, only needed if DB is live
    except ImportError:
        return None

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        query = f"""
            SELECT date, rate, bunker_price
            FROM {FREIGHT_RATES_TABLE}
            WHERE route = %s
            ORDER BY date ASC
        """
        df = pd.read_sql(query, conn, params=(route,))
        conn.close()

        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"])
        return df

    except Exception:
        # DB not reachable, table not seeded yet, wrong credentials, etc.
        # Silent fallback is intentional here — see generate_synthetic_series.
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
    but reproducible series.
    """
    rng = np.random.default_rng(seed + abs(hash(route)) % 10_000)

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
    Main entry point. Returns DataFrame[date, rate, bunker_price] for a route,
    sourced from MySQL when available, else synthetic.
    """
    df = _try_load_from_mysql(route)
    if df is not None:
        return df

    return generate_synthetic_series(route)


if __name__ == "__main__":
    for r in ["C5", "C3"]:
        d = load_freight_rate_history(r)
        print(f"{r}: {len(d)} rows, {d['date'].min().date()} -> {d['date'].max().date()}")
        print(d.tail(3), "\n")
