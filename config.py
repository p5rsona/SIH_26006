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
    "database": os.getenv("FR_DB_NAME", "maritime_data"),
}

# Expected table (per Person 1's schema): freight_rates_history
# Expected columns: date, route, rate, bunker_price (optional)
FREIGHT_RATES_TABLE = "freight_rates_history"

# Routes we forecast. Keep in sync with Person 1's `routes` table.
SUPPORTED_ROUTES = [
    "C5",       # Australia-China (coal)
    "C3",       # Brazil-China (iron ore)
    "P1A_82",   # Pacific round voyage
    "TD3C",     # Middle East-China (crude, tanker)
]

# Monsoon months relevant to east-coast India ports (affects laycan/draft
# and therefore freight rates on India-linked routes) — June to September.
MONSOON_MONTHS = {6, 7, 8, 9}

RANDOM_SEED = 42
