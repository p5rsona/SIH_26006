"""
Config for the Freight Rate Forecasting module (Person 2).

The database is Supabase (Postgres) — the connection itself lives in `db.py`
and `.env`, not here. If Supabase isn't reachable (or hasn't been seeded
yet), data_loader.py falls back to the CSVs below and then to synthetic data,
so this module can be developed/tested independently.
"""

import os

# Person 1's table (see schema_postgres.sql): freight_rates_history
# Columns: rate_date, index_name (BDI/BCI/BPI/BSI), index_value — daily, business days
FREIGHT_RATES_TABLE = "freight_rates_history"

_HERE = os.path.dirname(os.path.abspath(__file__))

# Local copy of the same table, used when Supabase isn't reachable
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

# The DB stores Baltic *index* history, not per-route rates, so each
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

# Weather / cyclone risk (see weather.py for the full model + real-vs-
# synthetic breakdown). Kept here too since optimizer.py and
# person5_risk_simulation.py both read them, same pattern as MONSOON_MONTHS.
CYCLONE_HARD_BLOCK_THRESHOLD = 0.75   # risk score above this hard-blocks a port/laycan combo
DEMURRAGE_RATE_USD_PER_DAY = 12000    # used to price expected weather delay into cost

RANDOM_SEED = 42
