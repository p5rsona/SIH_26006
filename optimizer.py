from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List

import pandas as pd
from ortools.sat.python import cp_model

import weather


OUTPUT_COLUMNS = [
    "cargo",
    "vessel",
    "port",
    "qty",
    "cost_breakdown"
]


def _to_ordinal(value: str) -> int:
    return date.fromisoformat(str(value)).toordinal()


def _overlap(a_start, a_end, b_start, b_end):
    return max(
        _to_ordinal(a_start),
        _to_ordinal(b_start)
    ) <= min(
        _to_ordinal(a_end),
        _to_ordinal(b_end)
    )


def optimize(
    cargo_list: List[Dict[str, Any]],
    vessel_list: List[Dict[str, Any]],
    constraints: Dict[str, Any],
    weather_multiplier: float = 1.0
) -> pd.DataFrame:
    """
    weather_multiplier: scales the expected weather/cyclone-delay cost
    (see weather.py). 1.0 = climatological expectation. Person 5's Monte
    Carlo simulation samples this per-scenario so realized weather outcomes
    vary around that expectation; leave it at 1.0 for a single "expected
    case" run.
    """

    ports = constraints["ports"]
    budget = constraints.get("budget")

    # Pre-compute each (cargo, port) pair's expected weather-risk cost and
    # whether that pair should be hard-blocked (cyclone risk too high for
    # that laycan window at that port), once — not inside the per-vessel
    # loops below.
    weather_cost = {}
    weather_blocked = {}
    for cargo in cargo_list:
        for port in ports:
            key = (cargo["cargo"], port["port"])
            weather_cost[key] = weather.weather_cost_adder(
                port["port"], cargo["laycan_start"], cargo["laycan_end"],
                weather_multiplier=weather_multiplier,
            )
            weather_blocked[key] = weather.is_extreme_risk(
                port["port"], cargo["laycan_start"], cargo["laycan_end"],
            )

    # Optional per-voyage economics from fleet_data.py:
    #   {(cargo, vessel, port): {"freight_per_tonne": ..., "voyage_fixed": ...}}
    # Freight depends on the route sailed and the repositioning (ballast) cost
    # depends on where the vessel is open, so neither is really one number per
    # vessel. When no matrix is given we fall back to the vessel's own
    # freight_cost_per_tonne / fixed_cost, so existing callers are unaffected.
    cost_matrix = constraints.get("cost_matrix") or {}

    def voyage_costs(cargo, vessel, port):
        entry = cost_matrix.get((cargo["cargo"], vessel["vessel"], port["port"]))
        if entry is None:
            return (
                float(vessel["freight_cost_per_tonne"]),
                float(vessel["fixed_cost"]),
            )
        return float(entry["freight_per_tonne"]), float(entry["voyage_fixed"])

    model = cp_model.CpModel()

    # -------------------------------------------------
    # DECISION VARIABLES
    # x[cargo, vessel, port] = 1 if selected
    # -------------------------------------------------

    x = {}

    for cargo in cargo_list:

        for vessel in vessel_list:

            for port in ports:

                key = (
                    cargo["cargo"],
                    vessel["vessel"],
                    port["port"]
                )

                x[key] = model.NewBoolVar(
                    f"x_{cargo['cargo']}_{vessel['vessel']}_{port['port']}"
                )

                # Port draft restriction
                if vessel["draft"] > port["max_draft"]:
                    model.Add(x[key] == 0)

                # Weather / cyclone hard block — this laycan window's
                # worst-day risk at this port is too high to route cargo
                # there at all (see weather.CYCLONE_HARD_BLOCK_THRESHOLD)
                if weather_blocked[(cargo["cargo"], port["port"])]:
                    model.Add(x[key] == 0)

                # Port size restriction (largest vessel the port can berth)
                if port.get("max_dwt_capable") and vessel["dwt"] > port["max_dwt_capable"]:
                    model.Add(x[key] == 0)

                # Laycan / availability restriction
                if not _overlap(
                    cargo["laycan_start"],
                    cargo["laycan_end"],
                    vessel["available_start"],
                    vessel["available_end"]
                ):
                    model.Add(x[key] == 0)

                # Optional allowed-port restriction
                if (
                    cargo.get("allowed_ports")
                    and port["port"] not in cargo["allowed_ports"]
                ):
                    model.Add(x[key] == 0)

    # -------------------------------------------------
    # CONSTRAINT 1:
    # Every cargo must be assigned exactly once
    # -------------------------------------------------

    for cargo in cargo_list:

        assignments = []

        for vessel in vessel_list:
            for port in ports:

                assignments.append(
                    x[
                        cargo["cargo"],
                        vessel["vessel"],
                        port["port"]
                    ]
                )

        model.Add(sum(assignments) == 1)

    # -------------------------------------------------
    # CONSTRAINT 2:
    # Vessel DWT capacity
    # -------------------------------------------------

    for vessel in vessel_list:

        model.Add(
            sum(
                x[
                    cargo["cargo"],
                    vessel["vessel"],
                    port["port"]
                ] * int(cargo["quantity"])

                for cargo in cargo_list
                for port in ports
            )
            <= int(vessel["dwt"])
        )

    # -------------------------------------------------
    # CONSTRAINT 3:
    # A vessel can only be on one voyage at a time. If two cargoes have
    # overlapping laycan windows, the same vessel cannot be assigned to
    # both — the DWT check above only limits total tonnage, it doesn't
    # stop the solver from "using" one ship for two simultaneous voyages.
    # -------------------------------------------------

    for vessel in vessel_list:

        for i, cargo_a in enumerate(cargo_list):

            for cargo_b in cargo_list[i + 1:]:

                if not _overlap(
                    cargo_a["laycan_start"],
                    cargo_a["laycan_end"],
                    cargo_b["laycan_start"],
                    cargo_b["laycan_end"]
                ):
                    continue

                vessel_for_a = sum(
                    x[cargo_a["cargo"], vessel["vessel"], port["port"]]
                    for port in ports
                )

                vessel_for_b = sum(
                    x[cargo_b["cargo"], vessel["vessel"], port["port"]]
                    for port in ports
                )

                model.Add(vessel_for_a + vessel_for_b <= 1)

    # -------------------------------------------------
    # COST CALCULATION
    # -------------------------------------------------

    total_cost_terms = []

    for cargo in cargo_list:

        quantity = int(cargo["quantity"])

        procurement_price = int(
            round(
                float(
                    cargo.get(
                        "procurement_price_per_tonne",
                        0
                    )
                )
            )
        )

        for vessel in vessel_list:

            for port in ports:

                variable = x[
                    cargo["cargo"],
                    vessel["vessel"],
                    port["port"]
                ]

                freight, fixed = voyage_costs(cargo, vessel, port)

                freight_cost = int(round(freight))

                fixed_cost = int(round(fixed))

                port_cost = int(
                    round(
                        float(port["port_cost"])
                    )
                )

                weather_risk_cost = int(
                    round(
                        weather_cost[(cargo["cargo"], port["port"])]
                    )
                )

                choice_cost = (
                    quantity
                    * (procurement_price + freight_cost)
                    + fixed_cost
                    + port_cost
                    + weather_risk_cost
                )

                total_cost_terms.append(
                    variable * choice_cost
                )

    total_cost = sum(total_cost_terms)

    # -------------------------------------------------
    # CONSTRAINT 4:
    # Budget
    # -------------------------------------------------

    if budget is not None:

        model.Add(
            total_cost <= int(round(float(budget)))
        )

    # -------------------------------------------------
    # OBJECTIVE:
    # MINIMIZE LANDED COST
    # -------------------------------------------------

    model.Minimize(total_cost)

    # -------------------------------------------------
    # SOLVER
    # -------------------------------------------------

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = 10
    # num_search_workers is deprecated in OR-Tools; num_workers replaces it.
    # Small models (the usual cargo x vessel x port sizes here) solve fastest
    # single-threaded — this matters because Person 5's Monte Carlo calls
    # optimize() hundreds/thousands of times.
    solver.parameters.num_workers = 1 if len(x) <= 500 else 8

    status = solver.Solve(model)

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE
    ):
        raise RuntimeError(
            "No feasible solution found."
        )

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------

    results = []

    for cargo in cargo_list:

        for vessel in vessel_list:

            for port in ports:

                key = (
                    cargo["cargo"],
                    vessel["vessel"],
                    port["port"]
                )

                if solver.Value(x[key]) == 1:

                    quantity = int(cargo["quantity"])

                    procurement = float(
                        cargo.get(
                            "procurement_price_per_tonne",
                            0
                        )
                    )

                    freight, fixed = voyage_costs(cargo, vessel, port)

                    port_cost = float(
                        port["port_cost"]
                    )

                    weather_risk_cost = weather_cost[
                        (cargo["cargo"], port["port"])
                    ]

                    breakdown = {

                        "procurement_cost":
                            round(
                                quantity * procurement,
                                2
                            ),

                        "freight_cost":
                            round(
                                quantity * freight,
                                2
                            ),

                        "vessel_fixed_cost":
                            round(
                                fixed,
                                2
                            ),

                        "port_cost":
                            round(
                                port_cost,
                                2
                            ),

                        "weather_risk_cost":
                            round(
                                weather_risk_cost,
                                2
                            ),

                        "total_cost":
                            round(
                                quantity
                                * (procurement + freight)
                                + fixed
                                + port_cost
                                + weather_risk_cost,
                                2
                            )
                    }

                    results.append({

                        "cargo":
                            cargo["cargo"],

                        "vessel":
                            vessel["vessel"],

                        "port":
                            port["port"],

                        "qty":
                            quantity,

                        "cost_breakdown":
                            json.dumps(
                                breakdown
                            )
                    })

    return pd.DataFrame(
        results,
        columns=OUTPUT_COLUMNS
    )


# =====================================================
# TEST DATA
# =====================================================

if __name__ == "__main__":

    from sample_data import get_sample_data

    cargo_list, vessel_list, constraints = (
        get_sample_data()
    )

    result = optimize(
        cargo_list,
        vessel_list,
        constraints
    )

    print("\n")
    print("=" * 60)
    print("OPTIMAL CHARTERING PLAN")
    print("=" * 60)

    print(
        result.to_string(index=False)
    )