"""
Seed data generator for the SIH Maritime Procurement/Chartering DB.
Updated to include Gangavaram, Gopalpur, Dhamra, Sagar, Haldia, Mozambique, Indonesia, Russia, and Capesize.
"""

import csv
import random
import math
from datetime import date, timedelta
import os

random.seed(42)  # reproducible

OUT = "/home/claude/sih_db/seed_csv"
os.makedirs(OUT, exist_ok=True)

HORIZON_DAYS = 540  
START_DATE = date(2025, 3, 1)

# ------------------------------------------------------------
# 1. PORTS  (Updated with 5 new ports & missing physical constraints)
# ------------------------------------------------------------
ports = [
    dict(port_id=1, port_name="Visakhapatnam", country="India", latitude=17.6868, longitude=83.2185, max_draft_m=14.0, loa_m=260.0, beam_m=43.0, num_berths=30, max_dwt_capable=95000, handling_rate_tpd=25000, notes="West Quay-1 max draft 14.0m"),
    dict(port_id=2, port_name="Paradip", country="India", latitude=20.2654, longitude=86.6763, max_draft_m=16.0, loa_m=300.0, beam_m=48.0, num_berths=21, max_dwt_capable=118000, handling_rate_tpd=35000, notes="Handles Capesize up to 118,000 DWT"),
    dict(port_id=3, port_name="Krishnapatnam", country="India", latitude=14.2500, longitude=80.1167, max_draft_m=18.5, loa_m=320.0, beam_m=50.0, num_berths=26, max_dwt_capable=180000, handling_rate_tpd=45000, notes="Deepest port; handles largest bulk carriers"),
    dict(port_id=4, port_name="Kakinada", country="India", latitude=16.9750, longitude=82.2790, max_draft_m=12.0, loa_m=230.0, beam_m=32.0, num_berths=4, max_dwt_capable=50000, handling_rate_tpd=15000, notes="Panamax/Supramax only"),
    dict(port_id=5, port_name="Gangavaram", country="India", latitude=17.6225, longitude=83.2422, max_draft_m=21.0, loa_m=300.0, beam_m=50.0, num_berths=9, max_dwt_capable=200000, handling_rate_tpd=40000, notes="Capable of fully laden Capesize"),
    dict(port_id=6, port_name="Gopalpur", country="India", latitude=19.3000, longitude=84.9750, max_draft_m=14.5, loa_m=260.0, beam_m=43.0, num_berths=3, max_dwt_capable=80000, handling_rate_tpd=20000, notes="All-weather deep water port"),
    dict(port_id=7, port_name="Dhamra", country="India", latitude=20.8250, longitude=86.9750, max_draft_m=18.0, loa_m=320.0, beam_m=50.0, num_berths=5, max_dwt_capable=180000, handling_rate_tpd=50000, notes="Deep draft port"),
    dict(port_id=8, port_name="Sagar-Sandheads", country="India", latitude=21.6500, longitude=88.0333, max_draft_m=10.5, loa_m=220.0, beam_m=32.0, num_berths=0, max_dwt_capable=40000, handling_rate_tpd=10000, notes="Anchorage point / lighterage"),
    dict(port_id=9, port_name="Haldia", country="India", latitude=22.0250, longitude=88.0620, max_draft_m=8.5, loa_m=230.0, beam_m=32.0, num_berths=14, max_dwt_capable=40000, handling_rate_tpd=15000, notes="Riverine port, draft restricted"),
]

# ------------------------------------------------------------
# 2. ORIGIN PORTS (Updated with Mozambique, Indonesia, Russia)
# ------------------------------------------------------------
origin_ports = [
    dict(origin_id=1, origin_name="Newcastle", country="Australia", commodity_focus="coal"),
    dict(origin_id=2, origin_name="Richards Bay", country="South Africa", commodity_focus="coal"),
    dict(origin_id=3, origin_name="US Gulf (New Orleans)", country="USA", commodity_focus="grain"),
    dict(origin_id=4, origin_name="Maputo", country="Mozambique", commodity_focus="coal"),
    dict(origin_id=5, origin_name="Samarinda", country="Indonesia", commodity_focus="coal"),
    dict(origin_id=6, origin_name="Novorossiysk", country="Russia", commodity_focus="grain"),
]

# ------------------------------------------------------------
# 3. ROUTES (Dynamically calculated based on base distance)
# ------------------------------------------------------------
routes = []
base_distances = {
    "Newcastle": 5100, "Richards Bay": 4300, "US Gulf (New Orleans)": 10800,
    "Maputo": 4100, "Samarinda": 2400, "Novorossiysk": 5500
}

rid = 1
for o in origin_ports:
    for p in ports:
        # Synthetic variation logic for distance based on latitude proxy
        variance = (p["port_id"] - 1) * 45
        dist = base_distances[o["origin_name"]] + variance if o["origin_name"] in ["Samarinda", "Newcastle"] else base_distances[o["origin_name"]] + (9 - p["port_id"]) * 45
        
        routes.append(dict(
            route_id=rid, origin_id=o["origin_id"], port_id=p["port_id"],
            distance_nm=dist, typical_transit_days=round(dist / (12 * 24), 1)
        ))
        rid += 1

# ------------------------------------------------------------
# 4. VESSELS (Added Capesize class)
# ------------------------------------------------------------
vessel_classes = {
    "Capesize":  dict(dwt=(110000, 180000), speed=(13.5, 15.0), cons_laden=(45, 55), cons_ballast=(35, 45)),
    "Panamax":   dict(dwt=(65000, 82000), speed=(12.5, 14.5), cons_laden=(28, 34), cons_ballast=(24, 29)),
    "Supramax":  dict(dwt=(50000, 60000), speed=(13.0, 14.5), cons_laden=(24, 29), cons_ballast=(20, 25)),
    "Handysize": dict(dwt=(28000, 40000), speed=(12.0, 14.0), cons_laden=(18, 23), cons_ballast=(15, 19)),
}
vessels = []
N_VESSELS = 30
for i in range(1, N_VESSELS + 1):
    vclass = random.choices(list(vessel_classes.keys()), weights=[0.20, 0.40, 0.30, 0.10])[0]
    spec = vessel_classes[vclass]
    open_port = random.choice(ports)["port_id"]
    open_offset = random.randint(0, HORIZON_DAYS - 30)
    vessels.append(dict(
        vessel_id=i,
        vessel_name=f"MV {vclass[:4].upper()}-{i:03d}",
        vessel_class=vclass,
        dwt=random.randint(*spec["dwt"]),
        speed_knots=round(random.uniform(*spec["speed"]), 1),
        consumption_tpd_laden=round(random.uniform(*spec["cons_laden"]), 1),
        consumption_tpd_ballast=round(random.uniform(*spec["cons_ballast"]), 1),
        open_port_id=open_port,
        open_date=(START_DATE + timedelta(days=open_offset)).isoformat(),
    ))

# ------------------------------------------------------------
# 5. FREIGHT RATES HISTORY
# ------------------------------------------------------------
def synth_series(n_days, start_val, target_end_val, vol, floor_val, seed_offset):
    rnd = random.Random(42 + seed_offset)
    vals = [start_val]
    drift = (target_end_val - start_val) / n_days
    for d in range(1, n_days):
        seasonal = 1 + 0.05 * math.sin(2 * math.pi * d / 365)
        shock = rnd.gauss(0, vol)
        mean_revert = 0.02 * (target_end_val - vals[-1])
        nxt = vals[-1] + drift + mean_revert + shock
        nxt = max(floor_val, nxt * seasonal / (1 + 0.05 * math.sin(2 * math.pi * (d - 1) / 365)))
        vals.append(round(nxt, 1))
    return vals

freight_rows = []
dates = [START_DATE + timedelta(days=d) for d in range(HORIZON_DAYS)]

bdi_series = synth_series(HORIZON_DAYS, 1900, 3200, 45, 900, 1)
bci_series = synth_series(HORIZON_DAYS, 3200, 5600, 110, 1200, 2)
bpi_series = synth_series(HORIZON_DAYS, 1500, 2430, 35, 700, 3)
bsi_series = synth_series(HORIZON_DAYS, 1100, 1660, 25, 600, 4)

for i, d in enumerate(dates):
    if d.weekday() >= 5: continue
    freight_rows.append(dict(rate_date=d.isoformat(), index_name="BDI", index_value=bdi_series[i]))
    freight_rows.append(dict(rate_date=d.isoformat(), index_name="BCI", index_value=bci_series[i]))
    freight_rows.append(dict(rate_date=d.isoformat(), index_name="BPI", index_value=bpi_series[i]))
    freight_rows.append(dict(rate_date=d.isoformat(), index_name="BSI", index_value=bsi_series[i]))

# ------------------------------------------------------------
# 6. COMMODITY PRICES
# ------------------------------------------------------------
coal_series = synth_series(HORIZON_DAYS, 115, 105, 2.5, 70, 11)
wheat_series = synth_series(HORIZON_DAYS, 230, 225, 4.0, 150, 12)
corn_series = synth_series(HORIZON_DAYS, 215, 218, 3.5, 140, 13)

commodity_rows = []
for i, d in enumerate(dates):
    if d.weekday() >= 5: continue
    commodity_rows.append(dict(price_date=d.isoformat(), commodity="coal_newcastle", origin_id=1, price_usd_per_tonne=coal_series[i]))
    commodity_rows.append(dict(price_date=d.isoformat(), commodity="wheat_gulf", origin_id=3, price_usd_per_tonne=wheat_series[i]))
    commodity_rows.append(dict(price_date=d.isoformat(), commodity="corn_gulf", origin_id=3, price_usd_per_tonne=corn_series[i]))

# ------------------------------------------------------------
# 7. FIXTURES
# ------------------------------------------------------------
fixtures = []
route_by_id = {r["route_id"]: r for r in routes}
bdi_by_date = {d.isoformat(): bdi_series[i] for i, d in enumerate(dates)}
commodity_price_by_date = {
    "coal_newcastle": {d.isoformat(): coal_series[i] for i, d in enumerate(dates)},
    "wheat_gulf": {d.isoformat(): wheat_series[i] for i, d in enumerate(dates)},
    "corn_gulf": {d.isoformat(): corn_series[i] for i, d in enumerate(dates)},
}

N_FIXTURES = 250
for fid in range(1, N_FIXTURES + 1):
    v = random.choice(vessels)
    r = random.choice(routes)
    op = origin_ports[r["origin_id"] - 1]
    commodity = "coal_newcastle" if op["commodity_focus"] == "coal" else random.choice(["wheat_gulf", "corn_gulf"])
    
    fdate = START_DATE + timedelta(days=random.randint(0, HORIZON_DAYS - 40))
    fdate_iso = fdate.isoformat()
    while fdate_iso not in bdi_by_date:  
        fdate += timedelta(days=1)
        fdate_iso = fdate.isoformat()

    bdi_val = bdi_by_date[fdate_iso]
    freight_rate = round(4 + (bdi_val / 1000) * 2.2 + r["distance_nm"] / 3000, 2)
    fob_price = commodity_price_by_date[commodity][fdate_iso]
    
    max_qty = min(v["dwt"], 180000)
    min_qty = min(40000, max_qty - 1000) if max_qty > 5000 else max_qty
    qty = random.randint(min_qty, max_qty)
    
    laycan_start = fdate + timedelta(days=random.randint(5, 15))
    laycan_end = laycan_start + timedelta(days=random.randint(3, 7))
    total_cost = round(qty * (freight_rate + fob_price), 2)

    fixtures.append(dict(
        fixture_id=fid, fixture_date=fdate_iso, vessel_id=v["vessel_id"], route_id=r["route_id"],
        commodity=commodity, qty_tonnes=qty, freight_rate_usd_per_tonne=freight_rate,
        fob_price_usd_per_tonne=fob_price, laycan_start=laycan_start.isoformat(),
        laycan_end=laycan_end.isoformat(), total_cost_usd=total_cost,
    ))

# ------------------------------------------------------------
# WRITE CSVs
# ------------------------------------------------------------
def write_csv(filename, rows, fieldnames):
    path = os.path.join(OUT, filename)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows):>6} rows -> {path}")

write_csv("ports.csv", ports, list(ports[0].keys()))
write_csv("origin_ports.csv", origin_ports, list(origin_ports[0].keys()))
write_csv("routes.csv", routes, list(routes[0].keys()))
write_csv("vessels.csv", vessels, list(vessels[0].keys()))
write_csv("freight_rates_history.csv", freight_rows, list(freight_rows[0].keys()))
write_csv("commodity_prices.csv", commodity_rows, list(commodity_rows[0].keys()))
write_csv("fixtures.csv", fixtures, list(fixtures[0].keys()))

print("\nDone. All CSVs in", OUT)