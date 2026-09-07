# Data Dictionary — SIH Maritime Procurement & Chartering Optimizer

Database: `sih_shipping` (MySQL). Schema in `schema.sql`. Seed data in `seed_csv/`.
Load with `python3 load_to_mysql.py --host <host> --user <user> --password <pw>`.

**Scope**: Coal (Newcastle, Richards Bay) and grain (US Gulf) cargo shipped by
Panamax/Supramax bulk carriers to 4 east-coast India ports (Visakhapatnam, Paradip,
Krishnapatnam, Kakinada), 3–6 month planning horizon.

---

## What's REAL vs SYNTHETIC (be ready to explain this to judges)

| Data | Status | Source |
|---|---|---|
| Port names, coordinates, draft limits, berth counts | **Real** | Port authority sites, Wikipedia, trade press (Sept 2026) |
| Origin port names (Newcastle, Richards Bay, US Gulf) | **Real** | Standard global coal/grain load ports |
| Route distances (nm) | **Real** (approx great-circle) | Standard maritime distance tables |
| Freight rate index anchors (current BDI/BCI/BPI/BSI levels, 52-wk range) | **Real** | Baltic Exchange via Trading Economics, Sept 2026 |
| Full daily freight rate history (18 months) | **Synthetic** | Random walk + mean reversion, calibrated to real anchor levels/volatility above |
| Commodity price anchors (Newcastle coal, Gulf wheat/corn levels) | **Real** | IEA Coal Mid-Year Update 2025, USDA AMS Gulf export bids, Sept 2026 |
| Full daily commodity price history | **Synthetic** | Same calibration approach as freight rates |
| Vessel fleet (24 vessels: DWT, speed, consumption, open dates) | **Synthetic** | Generated within realistic Panamax/Supramax/Handysize ranges |
| Fixtures (180 simulated past charter deals) | **Synthetic** | Derived from synthetic freight/price series + random vessel/route assignment |

**Talking point for judges**: real-world Baltic Exchange tick data and Clarksons
fixture data are paywalled/subscription-only. We used real, publicly available
anchor values (current index levels, historical price ranges, actual port specs)
to calibrate realistic synthetic time series — this is standard practice for
prototyping when the "ground truth" feed is commercial.

---

## Tables

### `ports` (4 rows — real)
East coast India destination ports in scope.
| Column | Type | Notes |
|---|---|---|
| port_id | INT PK | |
| port_name | VARCHAR | Visakhapatnam, Paradip, Krishnapatnam, Kakinada |
| max_draft_m | DECIMAL | Real max allowable draft — constrains which vessels can berth |
| num_berths | INT | Real berth counts |
| max_dwt_capable | INT | Approx largest vessel DWT the port can realistically take — **use this as your optimizer's port-capacity constraint** |

### `origin_ports` (3 rows — real)
Load ports: Newcastle (AU, coal), Richards Bay (ZA, coal), US Gulf/New Orleans (USA, grain).

### `routes` (12 rows — real distances)
Every origin × destination pair (3 origins × 4 ports). `distance_nm` and
`typical_transit_days` (at ~12 knots average, incl. weather margin) feed directly
into freight cost and laycan feasibility calculations.

### `vessels` (24 rows — synthetic)
Fleet of Panamax (50%), Supramax (35%), Handysize (15%) vessels with DWT, speed,
fuel consumption (laden/ballast), and an `open_port_id` + `open_date` (when/where
the vessel becomes available to charter) — this is what your optimizer assigns to cargo.

### `freight_rates_history` (1,540 rows — synthetic, real-calibrated)
Daily (business days only) BDI/BCI/BPI/BSI index values.
**This is your target variable for Person 2's forecasting model.**
Query example: `SELECT rate_date, index_value FROM freight_rates_history WHERE index_name='BPI' ORDER BY rate_date`

### `commodity_prices` (1,155 rows — synthetic, real-calibrated)
Daily FOB prices in USD/tonne for `coal_newcastle`, `wheat_gulf`, `corn_gulf`.
**Target variable for Person 3's forecasting model.**

### `fixtures` (180 rows — synthetic)
Simulated historical charter deals: vessel + route + commodity + qty + rates + cost.
Useful for Person 5's baseline model ("what would a naive spot-charter strategy
have cost historically") and as a sanity-check dataset for the optimizer's output shape.

---

## Suggested interface functions (for Person 2, 3, 4 to build against)

```python
def forecast_freight_rate(route_id: int, horizon_weeks: int) -> "DataFrame[date, rate, ci_lower, ci_upper]"
def forecast_commodity_price(commodity: str, horizon_weeks: int) -> "DataFrame[date, price, ci_lower, ci_upper]"
def optimize(cargo_list, vessel_list, constraints) -> "DataFrame[cargo, vessel, port, qty, cost_breakdown]"
```

---

## Known limitations (mention proactively if judges ask)
- Freight/commodity time series are synthetic — real day-to-day figures will differ, but distributional properties (range, volatility) are calibrated to real data.
- Port draft/DWT limits are approximate — actual berth-by-berth limits vary and change with dredging.
- Route distances are straight-line great-circle approximations, not actual sailed routes (which follow shipping lanes).
