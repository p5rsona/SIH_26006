from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List

import pandas as pd
from ortools.sat.python import cp_model


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
    constraints: Dict[str, Any]
) -> pd.DataFrame:

    ports = constraints["ports"]
    budget = constraints.get("budget")

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

                freight_cost = int(
                    round(
                        float(
                            vessel["freight_cost_per_tonne"]
                        )
                    )
                )

                fixed_cost = int(
                    round(
                        float(vessel["fixed_cost"])
                    )
                )

                port_cost = int(
                    round(
                        float(port["port_cost"])
                    )
                )

                choice_cost = (
                    quantity
                    * (procurement_price + freight_cost)
                    + fixed_cost
                    + port_cost
                )

                total_cost_terms.append(
                    variable * choice_cost
                )

    total_cost = sum(total_cost_terms)

    # -------------------------------------------------
    # CONSTRAINT 3:
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
    solver.parameters.num_search_workers = 8

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

                    freight = float(
                        vessel["freight_cost_per_tonne"]
                    )

                    fixed = float(
                        vessel["fixed_cost"]
                    )

                    port_cost = float(
                        port["port_cost"]
                    )

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

                        "total_cost":
                            round(
                                quantity
                                * (procurement + freight)
                                + fixed
                                + port_cost,
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