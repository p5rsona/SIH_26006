"""
person5_risk_simulation.py — Person 5: Risk Simulation & Baseline Comparison

Public contract (per the team doc):

    simulate_scenarios(n_scenarios) -> cost_distribution_df
        columns: scenario_id, optimized_cost, baseline_cost, savings_pct

This module:
  1. Pulls uncertainty from Person 2's freight forecast (forecast.py) and
     Person 3's commodity forecasts (forecast_<commodity>.csv files).
  2. Monte Carlo-samples cargo procurement prices and vessel freight costs
     around those forecasts.
  3. Re-runs Person 4's optimizer (optimizer.py) on every sampled scenario.
  4. Also runs a naive "first feasible option, no optimization" baseline on
     the same scenarios, so we can report a genuine cost-savings distribution
     instead of a single point estimate.

ASSUMPTION TO FLAG WITH THE TEAM: optimizer.py currently takes a static
`freight_cost_per_tonne` per vessel — there's no existing function that
converts Person 2's route-level $/day freight rate forecast into a
per-vessel $/tonne cost. Until that conversion exists, this module applies
the freight forecast's *relative* volatility (%) as a shared multiplicative
shock to every vessel's freight_cost_per_tonne each scenario. That's a
reasonable stand-in for "how much would freight cost swing", but ask
Person 2/4 whether there's a real $/day -> $/tonne formula you should use
instead once one exists.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecast import forecast_freight_rate
from optimizer import optimize
from sample_data import _ensure_forecast_file, get_sample_data

ROOT = Path(__file__).resolve().parent


# Person 3's forecast CSVs are keyed by full commodity name; cargo dicts
# (see sample_data.py) use short commodity codes. Keep this in sync with
# whatever Person 1/3 finalize.
COMMODITY_FORECAST_FILE = {
    "coal": "forecast_coal_newcastle.csv",
    "wheat": "forecast_wheat_gulf.csv",
    "corn": "forecast_corn_gulf.csv",
}

DEFAULT_ROUTE = "C5"
DEFAULT_HORIZON_WEEKS = 8


# =====================================================================
# STEP 1 — Turn Person 2 & Person 3's forecasts into distributions
# =====================================================================

def _commodity_price_distribution(commodity: str) -> Tuple[float, float]:
    """Returns (mean_price, sigma) for a commodity from Person 3's forecast CSV."""
    filename = COMMODITY_FORECAST_FILE.get(commodity)
    if filename is None:
        raise ValueError(f"No forecast file mapped for commodity '{commodity}'")

    # regenerates Person 3's CSV if it's missing instead of crashing
    df = pd.read_csv(_ensure_forecast_file(filename))
    mean_price = float(df["price"].mean())
    # Treat ci_upper/ci_lower as a ~95% band -> back out sigma
    sigma = float(((df["ci_upper"] - df["ci_lower"]) / 2 / 1.96).mean())
    return mean_price, sigma


def _freight_relative_volatility(
    route: str = DEFAULT_ROUTE,
    horizon_weeks: int = DEFAULT_HORIZON_WEEKS,
) -> float:
    """Returns the freight forecast's average relative sigma (e.g. 0.04 = 4%),
    from Person 2's forecast_freight_rate(). Called ONCE (SARIMAX fitting is
    slow) — not inside the Monte Carlo loop."""
    fc = forecast_freight_rate(route, horizon_weeks=horizon_weeks)
    sigma = (fc["ci_upper"] - fc["ci_lower"]) / 2 / 1.96
    relative_sigma = (sigma / fc["rate"]).mean()
    return float(relative_sigma)


# =====================================================================
# STEP 2 — Sample one Monte Carlo scenario
# =====================================================================

def _sample_scenario(
    cargo_list: List[Dict[str, Any]],
    vessel_list: List[Dict[str, Any]],
    commodity_dists: Dict[str, Tuple[float, float]],
    freight_rel_sigma: float,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns perturbed (cargo_list, vessel_list) for one scenario."""

    scenario_cargo = copy.deepcopy(cargo_list)
    scenario_vessels = copy.deepcopy(vessel_list)

    # Perturb each cargo's procurement price around its commodity forecast.
    for cargo in scenario_cargo:
        mean_price, sigma = commodity_dists[cargo["commodity"]]
        sampled_price = rng.normal(mean_price, sigma)
        cargo["procurement_price_per_tonne"] = max(sampled_price, 1.0)

    # Freight market moves together — one shared shock applied to every
    # vessel's freight cost, not an independent draw per vessel.
    freight_shock = rng.normal(0.0, freight_rel_sigma)
    for vessel in scenario_vessels:
        base = vessel["freight_cost_per_tonne"]
        vessel["freight_cost_per_tonne"] = max(base * (1 + freight_shock), 1.0)

    return scenario_cargo, scenario_vessels


# =====================================================================
# STEP 3 — Naive baseline: first feasible vessel/port, no cost minimization
# =====================================================================

def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    a_start, a_end = pd.Timestamp(a_start), pd.Timestamp(a_end)
    b_start, b_end = pd.Timestamp(b_start), pd.Timestamp(b_end)
    return max(a_start, b_start) <= min(a_end, b_end)


def _cost_breakdown(cargo, vessel, port) -> float:
    qty = cargo["quantity"]
    procurement = cargo.get("procurement_price_per_tonne", 0)
    freight = vessel["freight_cost_per_tonne"]
    return qty * (procurement + freight) + vessel["fixed_cost"] + port["port_cost"]


def greedy_baseline(
    cargo_list: List[Dict[str, Any]],
    vessel_list: List[Dict[str, Any]],
    constraints: Dict[str, Any],
) -> Optional[float]:
    """"Always take the first workable option" strategy — no cost
    minimization, just the first vessel/port that satisfies draft, laycan
    and remaining capacity. This is the thing your optimized plan should
    beat. Returns None if a cargo can't be placed at all (infeasible).

    A vessel can only run one voyage at a time, so besides remaining DWT
    this also checks that a candidate cargo's laycan window doesn't
    overlap any laycan window already booked onto that same vessel."""

    ports = constraints["ports"]
    remaining_dwt = {v["vessel"]: v["dwt"] for v in vessel_list}
    vessel_bookings = {v["vessel"]: [] for v in vessel_list}  # booked laycan windows per vessel
    total_cost = 0.0

    for cargo in cargo_list:
        placed = False

        for vessel in vessel_list:
            if remaining_dwt[vessel["vessel"]] < cargo["quantity"]:
                continue

            if any(
                _overlaps(cargo["laycan_start"], cargo["laycan_end"], booked_start, booked_end)
                for booked_start, booked_end in vessel_bookings[vessel["vessel"]]
            ):
                continue  # vessel is already committed to an overlapping voyage

            for port in ports:
                if vessel["draft"] > port["max_draft"]:
                    continue
                if not _overlaps(
                    cargo["laycan_start"], cargo["laycan_end"],
                    vessel["available_start"], vessel["available_end"],
                ):
                    continue
                if cargo.get("allowed_ports") and port["port"] not in cargo["allowed_ports"]:
                    continue

                total_cost += _cost_breakdown(cargo, vessel, port)
                remaining_dwt[vessel["vessel"]] -= cargo["quantity"]
                vessel_bookings[vessel["vessel"]].append((cargo["laycan_start"], cargo["laycan_end"]))
                placed = True
                break

            if placed:
                break

        if not placed:
            return None  # baseline couldn't even find a feasible plan

    return total_cost


# =====================================================================
# STEP 4 — Public contract: simulate_scenarios()
# =====================================================================

def simulate_scenarios(
    n_scenarios: int = 500,
    cargo_list: Optional[List[Dict[str, Any]]] = None,
    vessel_list: Optional[List[Dict[str, Any]]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    route: str = DEFAULT_ROUTE,
    horizon_weeks: int = DEFAULT_HORIZON_WEEKS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Runs `n_scenarios` Monte Carlo draws of procurement/freight prices,
    re-solves Person 4's optimizer on each, and compares against a naive
    baseline on the *same* scenario so the comparison is apples-to-apples.

    Returns
    -------
    pd.DataFrame[scenario_id, optimized_cost, baseline_cost, savings_pct]
    """
    if cargo_list is None or vessel_list is None or constraints is None:
        cargo_list, vessel_list, constraints = get_sample_data()

    commodity_dists = {
        commodity: _commodity_price_distribution(commodity)
        for commodity in COMMODITY_FORECAST_FILE
    }
    freight_rel_sigma = _freight_relative_volatility(route, horizon_weeks)

    rng = np.random.default_rng(seed)
    rows = []

    for scenario_id in range(n_scenarios):
        scenario_cargo, scenario_vessels = _sample_scenario(
            cargo_list, vessel_list, commodity_dists, freight_rel_sigma, rng
        )

        try:
            result = optimize(scenario_cargo, scenario_vessels, constraints)
            optimized_cost = sum(
                json.loads(row)["total_cost"] for row in result["cost_breakdown"]
            )
        except RuntimeError:
            optimized_cost = np.nan  # optimizer found no feasible plan

        baseline_cost = greedy_baseline(scenario_cargo, scenario_vessels, constraints)
        if baseline_cost is None:
            baseline_cost = np.nan

        savings_pct = (
            (baseline_cost - optimized_cost) / baseline_cost * 100
            if baseline_cost and not np.isnan(optimized_cost) and not np.isnan(baseline_cost)
            else np.nan
        )

        rows.append({
            "scenario_id": scenario_id,
            "optimized_cost": optimized_cost,
            "baseline_cost": baseline_cost,
            "savings_pct": savings_pct,
        })

    return pd.DataFrame(rows)


def summarize(cost_distribution_df: pd.DataFrame) -> Dict[str, float]:
    """Headline numbers for the pitch slide."""
    df = cost_distribution_df.dropna()
    return {
        "n_valid_scenarios": len(df),
        "mean_optimized_cost": df["optimized_cost"].mean(),
        "mean_baseline_cost": df["baseline_cost"].mean(),
        "mean_savings_pct": df["savings_pct"].mean(),
        "p5_savings_pct": df["savings_pct"].quantile(0.05),
        "p95_savings_pct": df["savings_pct"].quantile(0.95),
        "prob_optimized_beats_baseline": (df["optimized_cost"] < df["baseline_cost"]).mean(),
        "worst_case_optimized_cost_p95": df["optimized_cost"].quantile(0.95),
    }


if __name__ == "__main__":
    print("Running Monte Carlo risk simulation...")
    dist_df = simulate_scenarios(n_scenarios=200)

    print("\nSample of scenario results:")
    print(dist_df.head(10).round(2).to_string(index=False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, value in summarize(dist_df).items():
        print(f"{key:35s}: {value:,.2f}" if isinstance(value, float) else f"{key:35s}: {value}")
