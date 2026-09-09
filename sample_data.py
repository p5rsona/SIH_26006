from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent


def forecast_mean(filename):

    df = pd.read_csv(
        ROOT / filename
    )

    return float(
        df["price"].mean()
    )


def get_sample_data():

    coal_price = forecast_mean(
        "forecast_coal_newcastle (1).csv"
    )

    wheat_price = forecast_mean(
        "forecast_wheat_gulf.csv"
    )

    corn_price = forecast_mean(
        "forecast_corn_gulf.csv"
    )

    # ---------------------------------------------
    # CARGO
    # ---------------------------------------------

    cargo_list = [

        {
            "cargo": "Coal_A",
            "commodity": "coal",
            "quantity": 40000,
            "laycan_start": "2026-08-24",
            "laycan_end": "2026-09-10",
            "procurement_price_per_tonne":
                coal_price
        },

        {
            "cargo": "Wheat_A",
            "commodity": "wheat",
            "quantity": 25000,
            "laycan_start": "2026-08-24",
            "laycan_end": "2026-09-12",
            "procurement_price_per_tonne":
                wheat_price
        },

        {
            "cargo": "Corn_A",
            "commodity": "corn",
            "quantity": 30000,
            "laycan_start": "2026-08-24",
            "laycan_end": "2026-09-15",
            "procurement_price_per_tonne":
                corn_price
        }
    ]

    # ---------------------------------------------
    # VESSELS
    # ---------------------------------------------

    vessel_list = [

        {
            "vessel": "Vessel_1",
            "dwt": 50000,
            "draft": 11,
            "available_start": "2026-08-20",
            "available_end": "2026-09-15",
            "freight_cost_per_tonne": 28,
            "fixed_cost": 150000
        },

        {
            "vessel": "Vessel_2",
            "dwt": 40000,
            "draft": 13,
            "available_start": "2026-08-22",
            "available_end": "2026-09-18",
            "freight_cost_per_tonne": 24,
            "fixed_cost": 100000
        },

        {
            "vessel": "Vessel_3",
            "dwt": 60000,
            "draft": 14,
            "available_start": "2026-08-24",
            "available_end": "2026-09-18",
            "freight_cost_per_tonne": 22,
            "fixed_cost": 200000
        }
    ]

    # ---------------------------------------------
    # PORTS
    # ---------------------------------------------

    ports = [

        {
            "port": "Paradip",
            "max_draft": 12,
            "port_cost": 150000
        },

        {
            "port": "Visakhapatnam",
            "max_draft": 14,
            "port_cost": 180000
        },

        {
            "port": "Kakinada",
            "max_draft": 13,
            "port_cost": 130000
        }
    ]

    # ---------------------------------------------
    # GENERAL CONSTRAINTS
    # ---------------------------------------------

    constraints = {

        "ports": ports,

        "budget": 30000000
    }

    return (
        cargo_list,
        vessel_list,
        constraints
    )