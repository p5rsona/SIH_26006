"""
fleet_data.py — builds optimizer inputs from Person 1's real tables.

`sample_data.py` hands the optimizer 3 hardcoded vessels and 3 hardcoded ports.
This module does the same job from the actual database — MySQL or Supabase,
whichever db.py is pointed at — falling back to the seed CSVs:

    get_fleet_inputs(...) -> (cargo_list, vessel_list, constraints)

...in exactly the format optimizer.optimize() expects, so the two are
interchangeable.

Three things the raw tables don't store, and how they are derived here:

1. freight cost per tonne — calibrated from the historical `fixtures` table:
   a linear fit of realised $/tonne on (freight index level, route distance).
   The index level used is Person 2's SARIMAX forecast for the relevant
   Baltic index over the laycan horizon, so the forecast now actually feeds
   the optimizer. Falls back to the last observed index if statsmodels
   isn't available.

2. voyage fixed cost — the ballast (empty) leg that repositions the vessel
   from where it is open to the load port:
   days x consumption_tpd_ballast x bunker price. This is why a ship that is
   already near the load port can win even if its $/tonne rate is higher.

3. port cost — now a real column (`ports.port_cost_usd`).

Because freight and ballast cost depend on the cargo/vessel/port combination,
they are passed to the optimizer as `constraints["cost_matrix"]` rather than as
one number per vessel (see optimizer.optimize).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

BUNKER_PRICE_USD_PER_TONNE = 550.0   # VLSFO, order-of-magnitude anchor
DEFAULT_BUDGET = 30_000_000

# Which Baltic index tracks each vessel class
CLASS_TO_INDEX = {
    "Capesize": "BCI",
    "Panamax": "BPI",
    "Supramax": "BSI",
    "Handysize": "BSI",
}

# Where each commodity loads (origin_ports.origin_id)
COMMODITY_ORIGIN = {
    "coal": 1,      # Newcastle
    "wheat": 3,     # US Gulf
    "corn": 3,      # US Gulf
}

# Person 3's forecast files, per commodity
COMMODITY_FORECAST_FILE = {
    "coal": "forecast_coal_newcastle.csv",
    "wheat": "forecast_wheat_gulf.csv",
    "corn": "forecast_corn_gulf.csv",
}


# =====================================================================
# Table loading: MySQL first, CSV fallback
# =====================================================================

@lru_cache(maxsize=16)
def load_table(name: str) -> pd.DataFrame:
    """Loads one of Person 1's tables from the database (MySQL or
    Supabase/Postgres — see db.py), else from its seed CSV."""
    from db import read_sql

    df = read_sql(f"SELECT * FROM {name}")
    if df is not None and not df.empty:
        df.attrs["source"] = "database"
        return df

    df = pd.read_csv(ROOT / f"{name}.csv")
    df.attrs["source"] = "csv"
    return df


# =====================================================================
# 1. Freight rate: $/tonne calibrated from historical fixtures
# =====================================================================

@lru_cache(maxsize=1)
def _freight_rate_model() -> Tuple[float, float, float, Dict[str, float]]:
    """Fits realised fixture rates on (index level, distance):

        rate_usd_per_tonne ~ a + b * index_level + c * distance_nm

    Returns (a, b, c, scale) where `scale[index_name]` converts that index
    onto the same footing as the index the fit was done on (BDI), since BCI,
    BPI and BSI sit at very different absolute levels.
    """
    fixtures = load_table("fixtures").copy()
    routes = load_table("routes")
    rates = load_table("freight_rates_history").copy()

    rates["rate_date"] = pd.to_datetime(rates["rate_date"])
    bdi = rates[rates["index_name"] == "BDI"].set_index("rate_date")["index_value"].astype(float)

    fixtures["fixture_date"] = pd.to_datetime(fixtures["fixture_date"])
    fixtures = fixtures.merge(routes[["route_id", "distance_nm"]], on="route_id", how="left")
    fixtures["index_level"] = fixtures["fixture_date"].map(bdi)
    fixtures = fixtures.dropna(subset=["index_level", "distance_nm", "freight_rate_usd_per_tonne"])

    X = np.column_stack([
        np.ones(len(fixtures)),
        fixtures["index_level"].astype(float),
        fixtures["distance_nm"].astype(float),
    ])
    y = fixtures["freight_rate_usd_per_tonne"].astype(float).values
    a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]

    means = rates.groupby("index_name")["index_value"].mean()
    scale = {name: float(means["BDI"] / means[name]) for name in means.index}
    return float(a), float(b), float(c), scale


def freight_usd_per_tonne(index_name: str, index_level: float, distance_nm: float) -> float:
    a, b, c, scale = _freight_rate_model()
    effective = index_level * scale.get(index_name, 1.0)
    return max(a + b * effective + c * distance_nm, 1.0)


@lru_cache(maxsize=8)
def forecast_index_level(index_name: str, horizon_weeks: int = 8) -> float:
    """Mean forecast level of a Baltic index over the horizon (Person 2's
    SARIMAX model). Falls back to the last observed value."""
    rates = load_table("freight_rates_history").copy()
    rates["rate_date"] = pd.to_datetime(rates["rate_date"])
    series = rates[rates["index_name"] == index_name][["rate_date", "index_value"]]
    series = series.rename(columns={"rate_date": "date", "index_value": "rate"})
    series["rate"] = series["rate"].astype(float)
    weekly = series.set_index("date")["rate"].resample("W-MON").mean().interpolate().dropna()

    try:
        from models import SarimaxModel

        model = SarimaxModel().fit(weekly.reset_index().rename(columns={"rate": "rate"}))
        return float(np.mean(model.predict(horizon_weeks).point))
    except Exception:
        return float(weekly.iloc[-1])


# =====================================================================
# 2. Ballast leg: cost of repositioning a vessel to the load port
# =====================================================================

def _ballast_cost(vessel: pd.Series, origin_id: int, routes: pd.DataFrame) -> float:
    """Fuel cost of sailing empty from the vessel's open port to the load port.
    Falls back to the average route distance when that pair isn't in `routes`."""
    match = routes[
        (routes["origin_id"] == origin_id) & (routes["port_id"] == vessel.get("open_port_id"))
    ]
    distance = float(match["distance_nm"].iloc[0]) if len(match) else float(routes["distance_nm"].mean())
    days = distance / (float(vessel["speed_knots"]) * 24.0)
    return days * float(vessel["consumption_tpd_ballast"]) * BUNKER_PRICE_USD_PER_TONNE


# =====================================================================
# 3. Cargoes, priced from Person 3's forecasts
# =====================================================================

def _commodity_price(commodity: str) -> float:
    from sample_data import _ensure_forecast_file

    path = _ensure_forecast_file(COMMODITY_FORECAST_FILE[commodity])
    return float(pd.read_csv(path)["price"].mean())


def default_cargoes(laycan_start: str, laycan_end: str) -> List[Dict[str, Any]]:
    return [
        {
            "cargo": name, "commodity": commodity, "quantity": qty,
            "laycan_start": laycan_start, "laycan_end": laycan_end,
            "origin_id": COMMODITY_ORIGIN[commodity],
            "procurement_price_per_tonne": _commodity_price(commodity),
        }
        for name, commodity, qty in [
            ("Coal_A", "coal", 40000), ("Wheat_A", "wheat", 25000), ("Corn_A", "corn", 30000)
        ]
    ]


# =====================================================================
# Main entry point
# =====================================================================

def get_fleet_inputs(
    laycan_start: str = "2026-08-24",
    laycan_end: str = "2026-09-15",
    cargo_list: Optional[List[Dict[str, Any]]] = None,
    budget: Optional[float] = DEFAULT_BUDGET,
    horizon_weeks: int = 8,
    max_vessels: int = 25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Returns (cargo_list, vessel_list, constraints) built from the real
    ports/vessels/routes/fixtures tables, ready for optimizer.optimize()."""
    ports_df = load_table("ports")
    vessels_df = load_table("vessels")
    routes_df = load_table("routes")

    if cargo_list is None:
        cargo_list = default_cargoes(laycan_start, laycan_end)

    window_start, window_end = pd.Timestamp(laycan_start), pd.Timestamp(laycan_end)

    # --- vessels open during the laycan window -----------------------
    vessels_df = vessels_df.copy()
    vessels_df["open_date"] = pd.to_datetime(vessels_df["open_date"])
    if "available_until" in vessels_df.columns:
        vessels_df["available_until"] = pd.to_datetime(vessels_df["available_until"])
    else:  # older seed data without the column
        vessels_df["available_until"] = vessels_df["open_date"] + pd.Timedelta(days=180)

    available = vessels_df[
        (vessels_df["open_date"] <= window_end) & (vessels_df["available_until"] >= window_start)
    ].head(max_vessels)

    if available.empty:
        raise RuntimeError(
            f"No vessels are open between {laycan_start} and {laycan_end}. "
            "Widen the laycan window or regenerate the fleet."
        )

    vessel_list = [
        {
            "vessel": row["vessel_name"],
            "dwt": int(row["dwt"]),
            "draft": float(row.get("draft_m") or 0),
            "available_start": max(row["open_date"], window_start).date().isoformat(),
            "available_end": row["available_until"].date().isoformat(),
            # placeholders: the real per-voyage numbers live in cost_matrix below
            "freight_cost_per_tonne": 0.0,
            "fixed_cost": 0.0,
            "vessel_class": row.get("vessel_class", ""),
        }
        for _, row in available.iterrows()
    ]

    ports = [
        {
            "port": row["port_name"],
            "max_draft": float(row["max_draft_m"]),
            "port_cost": float(row.get("port_cost_usd") or 150_000),
            "max_dwt_capable": int(row.get("max_dwt_capable") or 0),
        }
        for _, row in ports_df.iterrows()
    ]

    # --- per cargo x vessel x port voyage economics ------------------
    index_levels = {
        index: forecast_index_level(index, horizon_weeks)
        for index in set(CLASS_TO_INDEX.get(v.get("vessel_class", ""), "BDI") for v in vessel_list)
    }
    port_ids = dict(zip(ports_df["port_name"], ports_df["port_id"]))

    cost_matrix: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for cargo in cargo_list:
        origin_id = cargo.get("origin_id", COMMODITY_ORIGIN.get(cargo.get("commodity"), 1))
        for vessel, (_, row) in zip(vessel_list, available.iterrows()):
            index_name = CLASS_TO_INDEX.get(row.get("vessel_class", ""), "BDI")
            ballast = _ballast_cost(row, origin_id, routes_df)
            for port in ports:
                leg = routes_df[
                    (routes_df["origin_id"] == origin_id)
                    & (routes_df["port_id"] == port_ids.get(port["port"]))
                ]
                distance = float(leg["distance_nm"].iloc[0]) if len(leg) else float(routes_df["distance_nm"].mean())
                cost_matrix[(cargo["cargo"], vessel["vessel"], port["port"])] = {
                    "freight_per_tonne": freight_usd_per_tonne(index_name, index_levels[index_name], distance),
                    "voyage_fixed": ballast,
                }

    constraints = {"ports": ports, "budget": budget, "cost_matrix": cost_matrix}
    return cargo_list, vessel_list, constraints


if __name__ == "__main__":
    cargo_list, vessel_list, constraints = get_fleet_inputs()
    print(f"{len(cargo_list)} cargoes, {len(vessel_list)} vessels open, {len(constraints['ports'])} ports")
    print(pd.DataFrame(vessel_list)[["vessel", "vessel_class", "dwt", "draft", "available_start", "available_end"]].to_string(index=False))
    sample = list(constraints["cost_matrix"].items())[:3]
    for key, value in sample:
        print(key, {k: round(v, 1) for k, v in value.items()})

    from optimizer import optimize
    plan = optimize(cargo_list, vessel_list, constraints)
    print("\n" + plan.to_string(index=False))
