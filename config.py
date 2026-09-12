"""
Config for the Freight Rate Forecasting module (Person 2).

DB_CONFIG should match Person 1's MySQL schema. If the DB isn't reachable
(e.g. Person 1 hasn't seeded it yet), data_loader.py falls back to
synthetic data so this module can be developed/tested independently.
"""

import os

DB_CONFIG = {
    "host": os.getenv("FR_DB_HOST", "localhost"),
    "port": int(os.getenv("FR_DB_PORT", "3306")),
    "user": os.getenv("FR_DB_USER", "root"),
    "password": os.getenv("FR_DB_PASSWORD", ""),
    "database": os.getenv("FR_DB_NAME", "sih_shipping"),  # same DB load_to_mysql.py creates
}

# Person 1's table (see schema.sql): freight_rates_history
# Columns: rate_date, index_name (BDI/BCI/BPI/BSI), index_value — daily, business days
FREIGHT_RATES_TABLE = "freight_rates_history"

_HERE = os.path.dirname(os.path.abspath(__file__))

# Local copy of the same table, used when MySQL isn't reachable
FREIGHT_RATES_CSV = os.path.join(_HERE, "freight_rates_history.csv")

# Real public data written by fetch_real_data.py. When these exist they are
# preferred over the synthetic CSVs above.
FREIGHT_RATES_REAL_CSV = os.path.join(_HERE, "freight_rates_real.csv")
COMMODITY_PRICES_REAL_CSV = os.path.join(_HERE, "commodity_prices_real.csv")

# Routes we forecast. Keep in sync with Person 1's `routes` table.
SUPPORTED_ROUTES = [
    "C5",       # Australia-China (coal)
    "C3",       # Brazil-China (iron ore)
    "P1A_82",   # Pacific round voyage
    "TD3C",     # Middle East-China (crude, tanker)
    # Real USDA series (weekly ocean freight cost indices, 2002-) — these are
    # the only routes here backed by real published data.
    "USG_OCEAN",   # US Gulf ocean vessel rate  (grain)
    "PNW_OCEAN",   # Pacific Northwest ocean vessel rate (grain)
]

# Person 1's DB stores Baltic *index* history, not per-route rates, so each
# route is forecast from the index that tracks its vessel class. Routes
# mapped to None have no matching index in the data (TD3C is a tanker
# route; the DB only has dry-bulk indices) and use the synthetic fallback.
ROUTE_TO_INDEX = {
    "C5": "BCI",       # Capesize
    "C3": "BCI",       # Capesize
    "P1A_82": "BPI",   # Panamax
    "TD3C": None,      # tanker — no index in the dataset
    "USG_OCEAN": "USDA_GULF_VESSEL",    # real (USDA)
    "PNW_OCEAN": "USDA_PNW_VESSEL",     # real (USDA)
}

# Which index series come from real published data rather than the generator
REAL_INDEX_NAMES = {"USDA_GULF_VESSEL", "USDA_PNW_VESSEL", "USDA_BARGE"}

# Monsoon months relevant to east-coast India ports (affects laycan/draft
# and therefore freight rates on India-linked routes) — June to September.
MONSOON_MONTHS = {6, 7, 8, 9}

RANDOM_SEED = 42
