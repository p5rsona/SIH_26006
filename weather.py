"""
weather.py — Weather & Cyclone Risk Factor

Adds a weather/cyclone risk signal for the four east-coast India destination
ports and plugs into the rest of the pipeline the same way Person 1-5's
modules plug into each other:

  - features.py                 -> extra seasonal regressor for Person 2/3's
                                    forecasting models (`cyclone_season_weight`,
                                    alongside the existing `is_monsoon` flag)
  - optimizer.py                -> every (cargo, port) choice picks up an
                                    expected weather-risk cost (demurrage /
                                    diversion), and a port is hard-blocked for
                                    a laycan window if its cyclone risk is
                                    above CYCLONE_HARD_BLOCK_THRESHOLD
  - person5_risk_simulation.py  -> Monte Carlo draws a per-scenario
                                    "how bad did weather actually turn out"
                                    multiplier on top of the expected cost,
                                    so the cost distribution reflects that
                                    cyclone landfall is itself uncertain
  - dashboard.py                -> "Weather & Cyclone Risk" tab

WHAT'S REAL vs SYNTHETIC (same convention as data_dictionary.md)
------------------------------------------------------------------
| Data                                            | Status    | Source |
|--------------------------------------------------|-----------|--------|
| Bay of Bengal monthly cyclone-frequency SHAPE     | Real      | IMD / RSMC New Delhi cyclone e-Atlas (1891-2022) climatology: bimodal season, pre-monsoon Apr-May peak and a much larger post-monsoon Oct-Dec peak, sharpest in November (Bhardwaj & Singh 2020; NDMA "Cyclone" briefing) |
| Which coast stretch takes the most landfalls      | Real      | Odisha + north Andhra Pradesh (the Paradip/Visakhapatnam stretch) records more Bay of Bengal landfalls than the Nellore/Krishnapatnam stretch further south (RSMC New Delhi landfall climatology) |
| Exact per-port numeric exposure multipliers       | Synthetic | Calibrated to the real landfall pattern above — not an official per-port risk index |
| Per-event expected port-closure / queueing delay  | Synthetic | Ballpark from press reports of past India east-coast cyclone port closures (typically 1-4 days shut, longer queue-clearing) |
| Demurrage rate used to price the delay in dollars | Synthetic | Typical Panamax/Supramax demurrage ballpark, not a live quote |
| Live cyclone alerts                               | Optional  | `fetch_live_cyclone_alert()` will use a reachable IMD/GDACS-style feed if the deployment has internet access; this sandbox does not, so it always falls back to the climatology model below (same fallback style as data_loader.py's Supabase -> CSV -> synthetic chain) |

None of this replaces a real IMD warning feed for an actual voyage — it's a
planning-stage risk factor for the optimizer/forecast, not a safety system.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import CYCLONE_HARD_BLOCK_THRESHOLD, DEMURRAGE_RATE_USD_PER_DAY, MONSOON_MONTHS

# =====================================================================
# CONFIG — see the REAL vs SYNTHETIC table above for what's calibrated
# vs anchored to real climatology
# =====================================================================

# Relative monthly cyclone-genesis weight for the Bay of Bengal, real IMD
# climatological shape (bimodal: pre-monsoon Apr-May bump, much bigger
# post-monsoon Oct-Dec bump, sharpest in Nov), rescaled to [0, 1].
CYCLONE_MONTHLY_WEIGHT = {
    1: 0.05,   # Jan  — winter, cyclogenesis essentially dormant
    2: 0.03,   # Feb  — lowest point of the year
    3: 0.05,   # Mar  — starting to warm up
    4: 0.25,   # Apr  — pre-monsoon season begins
    5: 0.55,   # May  — pre-monsoon peak (Fani 2019, Amphan 2020, Nargis 2008)
    6: 0.15,   # Jun  — SW monsoon onset suppresses cyclogenesis...
    7: 0.10,   # Jul  — ...but monsoon depressions still roughen seas/ports
    8: 0.10,   # Aug
    9: 0.20,   # Sep  — monsoon withdrawal, disturbances start rebuilding
    10: 0.70,  # Oct  — post-monsoon season begins in earnest
    11: 1.00,  # Nov  — sharpest peak of the year (Phailin-adjacent, Odisha coast)
    12: 0.45,  # Dec  — tapering off but still active
}

# East-coast India destination ports (from ports.csv) and their relative
# cyclone-landfall exposure. Real anchor: Odisha/north-AP coast (Paradip,
# Visakhapatnam) records more Bay of Bengal landfalls than the Nellore/
# Krishnapatnam stretch further south; Kakinada sits in between, in the
# Godavari delta, which also sees frequent landfalls. Numeric values below
# are a synthetic index calibrated to that real ordering, not an official
# per-port score.
PORT_CYCLONE_EXPOSURE = {
    "Paradip": 1.00,          # Odisha coast — historically the most landfalls
    "Visakhapatnam": 0.90,    # North Andhra Pradesh — also high exposure
    "Kakinada": 0.85,         # Godavari delta — frequent landfalls, shallower approach
    "Krishnapatnam": 0.65,    # Southern AP — statistically fewer direct hits
}
DEFAULT_PORT_EXPOSURE = 0.75  # used for any port not in the table above

# During the SW monsoon (Jun-Sep), cyclogenesis itself is suppressed (see
# the monthly weights above) but rough seas / heavy rain still add a small
# baseline operational risk on top of the cyclone signal.
MONSOON_BASELINE_RISK = 0.15

# Planning-stage assumption (synthetic — see table above). Demurrage rate
# and the hard-block threshold live in config.py, alongside MONSOON_MONTHS,
# since optimizer.py and person5_risk_simulation.py both read them directly.
MAX_CYCLONE_DELAY_DAYS = 4.0  # expected port-closure + queue-clearing at max risk


def _month_weight(month: int) -> float:
    return CYCLONE_MONTHLY_WEIGHT.get(month, 0.05)


def _port_exposure(port_name: str) -> float:
    return PORT_CYCLONE_EXPOSURE.get(port_name, DEFAULT_PORT_EXPOSURE)


def daily_cyclone_risk(port_name: str, dt: "pd.Timestamp | date | str") -> float:
    """Risk score in [0, 1] for one port on one calendar day.

    Combines the seasonal cyclone-genesis weight with the port's landfall
    exposure, plus a small SW-monsoon baseline for rough-sea/port-congestion
    risk that isn't cyclone-specific.
    """
    ts = pd.Timestamp(dt)
    month = ts.month

    cyclone_component = _month_weight(month) * _port_exposure(port_name)
    monsoon_component = MONSOON_BASELINE_RISK if month in MONSOON_MONTHS else 0.0

    return float(np.clip(cyclone_component + monsoon_component, 0.0, 1.0))


def window_risk(port_name: str, start: "pd.Timestamp | date | str", end: "pd.Timestamp | date | str") -> float:
    """Worst-day risk score over a laycan/date window.

    Uses the max (not the average) because a single cyclone landfall day is
    enough to close a port and disrupt the whole window's plan — this is a
    hazard window, not something you can smooth over the average day.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts

    days = pd.date_range(start_ts, end_ts, freq="D")
    if len(days) == 0:
        days = [start_ts]

    return float(max(daily_cyclone_risk(port_name, d) for d in days))


def expected_delay_days(port_name: str, start, end, weather_multiplier: float = 1.0) -> float:
    """Expected weather-related delay (port closure + queue-clearing), in
    days, for a cargo/port/laycan-window combination."""
    risk = window_risk(port_name, start, end)
    return risk * MAX_CYCLONE_DELAY_DAYS * weather_multiplier


def weather_cost_adder(
    port_name: str,
    start,
    end,
    demurrage_rate: Optional[float] = None,
    weather_multiplier: float = 1.0,
) -> float:
    """Dollar cost to add to a (cargo, port) choice: expected delay days
    priced at a demurrage rate. This is what the optimizer minimizes over —
    it lets the solver trade off a cheaper-but-riskier port against a
    pricier-but-safer one instead of ignoring weather entirely."""
    rate = DEMURRAGE_RATE_USD_PER_DAY if demurrage_rate is None else demurrage_rate
    return expected_delay_days(port_name, start, end, weather_multiplier) * rate


def is_extreme_risk(port_name: str, start, end, threshold: Optional[float] = None) -> bool:
    """True if the laycan window's worst-day risk at this port is high
    enough that the plan shouldn't route cargo there at all (e.g. it's
    solidly inside the November peak at a high-exposure port)."""
    limit = CYCLONE_HARD_BLOCK_THRESHOLD if threshold is None else threshold
    return window_risk(port_name, start, end) >= limit


def sample_weather_multiplier(rng: np.random.Generator) -> float:
    """One Monte Carlo draw of 'how much worse/better than the climatological
    expectation did weather actually turn out this scenario'. Lognormal so
    it's always positive and mildly right-skewed (a bad cyclone season can
    blow the expected cost out a lot more than a mild season can undercut
    it), centered so E[multiplier] ~= 1."""
    sigma = 0.5
    mu = -0.5 * sigma**2  # keeps the mean at 1.0
    return float(rng.lognormal(mean=mu, sigma=sigma))


def monthly_climatology_table() -> pd.DataFrame:
    """Long-format table of (port, month, risk) for charting on the
    dashboard — one representative mid-month day per (port, month)."""
    rows = []
    for port_name in PORT_CYCLONE_EXPOSURE:
        for month in range(1, 13):
            sample_day = date(2026, month, 15)
            rows.append({
                "port": port_name,
                "month": month,
                "month_name": sample_day.strftime("%b"),
                "risk_score": daily_cyclone_risk(port_name, sample_day),
            })
    return pd.DataFrame(rows)


def fetch_live_cyclone_alert(port_name: str, timeout: float = 3.0) -> Optional[dict]:
    """Optional live-data hook. Tries a public cyclone-tracking feed; returns
    None (and the caller should fall back to the climatology model above) if
    it's unreachable — this sandbox has no general internet access, but a
    real deployment might. Mirrors data_loader.py's Supabase -> CSV -> synthetic
    fallback chain, just one step shorter."""
    try:
        import requests  # local import: optional dependency, only needed here
        # Placeholder public endpoint — swap in IMD/RSMC New Delhi's or
        # GDACS's actual alert feed URL once the team picks one.
        resp = requests.get("https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH",
                             params={"eventtype": "TC"}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


if __name__ == "__main__":
    print("Monthly cyclone-risk climatology by port (mid-month sample day):\n")
    table = monthly_climatology_table().pivot(index="month_name", columns="port", values="risk_score")
    table = table.reindex(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    print(table.round(2).to_string())

    print("\nExample: Paradip, 24 Aug - 10 Sep 2026 laycan window")
    print("  window_risk:      ", round(window_risk("Paradip", "2026-08-24", "2026-09-10"), 3))
    print("  expected_delay_d: ", round(expected_delay_days("Paradip", "2026-08-24", "2026-09-10"), 2))
    print("  weather_cost_add: $", round(weather_cost_adder("Paradip", "2026-08-24", "2026-09-10"), 2))

    print("\nExample: Paradip, 1-20 Nov 2026 laycan window (peak cyclone season)")
    print("  window_risk:      ", round(window_risk("Paradip", "2026-11-01", "2026-11-20"), 3))
    print("  is_extreme_risk:  ", is_extreme_risk("Paradip", "2026-11-01", "2026-11-20"))
