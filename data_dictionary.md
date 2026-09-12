# Data Dictionary — SIH Maritime Procurement & Chartering Optimizer

Database: `sih_shipping` (MySQL) or a Supabase/Postgres project — set `DB_BACKEND`
and load with `python load_to_db.py` (see README). `schema_postgres.sql` mirrors
`schema.sql` for Postgres; keep both in sync.
Original: Schema in `schema.sql`. Seed CSVs in the project folder (or `seed_csv/` if you create one).
Load with `python load_to_mysql.py --host <host> --user <user> --password <pw>`.

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
CSV note: `ports.csv` omits the nullable `loa_m`, `beam_m` and `handling_rate_tpd`
columns that `schema.sql` defines; they load as NULL.

East coast India destination ports in scope.
| Column | Type | Notes |
|---|---|---|
| port_id | INT PK | |
| port_name | VARCHAR | Visakhapatnam, Paradip, Krishnapatnam, Kakinada |
| max_draft_m | DECIMAL | Real max allowable draft — constrains which vessels can berth |
| num_berths | INT | Real berth counts |
| max_dwt_capable | INT | Approx largest vessel DWT the port can realistically take — enforced by the optimizer |
| port_cost_usd | DECIMAL | Fixed cost of one call (dues, pilotage, agency) — optimizer input. Synthetic: `60,000 + 0.9 x max_dwt_capable` |

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

| Column | Notes |
|---|---|
| draft_m | Laden draft, checked against `ports.max_draft_m`. Synthetic, from class + DWT (Handysize ~9.5-11 m, Supramax ~12-13, Panamax ~13.5-14.5, Capesize ~17-18.5) |
| available_until | End of the charter availability window: `open_date + 180 days` |

`fleet_data.py` turns these rows into optimizer inputs — see README. Six of the 24
vessels are open during the sample laycan window (2026-08-24 to 2026-09-15).

### `freight_rates_history` (1,540 rows — synthetic, real-calibrated)
Daily (business days only) BDI/BCI/BPI/BSI index values.
**This is your target variable for Person 2's forecasting model.**
Query example: `SELECT rate_date, index_value FROM freight_rates_history WHERE index_name='BPI' ORDER BY rate_date`

Note: this table holds **indices, not per-route rates**, so Person 2 maps each
route to the index for that vessel class (`config.ROUTE_TO_INDEX`): C5 and C3 ->
BCI (Capesize), P1A_82 -> BPI (Panamax). TD3C is a tanker route with no matching
index here, so it falls back to a synthetic series. `data_loader.py` averages the
daily index to weekly (W-MON) before modelling. Values are **index points, not
$/day or $/tonne** — converting an index level into a per-tonne freight cost for
the optimizer is still an open task.

### `commodity_prices` (1,155 rows — synthetic, real-calibrated)
Daily FOB prices in USD/tonne for `coal_newcastle`, `wheat_gulf`, `corn_gulf`.
**Target variable for Person 3's forecasting model.**

### `fixtures` (180 rows — synthetic)
Simulated historical charter deals: vessel + route + commodity + qty + rates + cost.
Useful for Person 5's baseline model ("what would a naive spot-charter strategy
have cost historically") and as a sanity-check dataset for the optimizer's output shape.

---

## Interface contracts (as implemented)

```python
# Person 2 - forecast.py
forecast_freight_rate(route: str, horizon_weeks: int, model="sarimax") -> DataFrame[date, rate, ci_lower, ci_upper]
forecast_with_disagreement(route: str, horizon_weeks: int)             -> + xgb_rate, disagreement, disagreement_pct

# Person 3 - person3_commodity_forecast.py
forecast_commodity_price(commodity: str, horizon_weeks: int = 4)       -> DataFrame[date, price, ci_lower, ci_upper]

# Person 4 - optimizer.py
optimize(cargo_list, vessel_list, constraints)                         -> DataFrame[cargo, vessel, port, qty, cost_breakdown]

# Person 5 - person5_risk_simulation.py
simulate_scenarios(n_scenarios: int)                                   -> DataFrame[scenario_id, optimized_cost, baseline_cost, savings_pct]
```
`route` is a route code from `config.SUPPORTED_ROUTES` (C5, C3, P1A_82, TD3C),
**not** a `routes.route_id` — reconciling the two is an open task. Commodity
forecasts take the commodity name only (`coal_newcastle`, `wheat_gulf`,
`corn_gulf`); the origin is implied by the commodity.

### Same contracts over HTTP (`api.py`)
```
GET  /forecast/freight?route=C5&weeks=8
GET  /forecast/commodity?commodity=coal_newcastle&weeks=4
POST /optimize          {"budget": 30000000}
POST /simulate          {"n_scenarios": 200}
GET  /health    GET /meta
```
Start it with `python -m uvicorn api:app --port 8000`; interactive docs at `/docs`.

---

## Regenerating the data
`python generate_data.py` rewrites all 7 CSVs in the project folder; reload the DB
afterwards with `python load_to_mysql.py --user root`. The current generator
produces a **larger world than the tables above**: 9 ports (adds Gangavaram,
Gopalpur, Dhamra, Sagar-Sandheads, Haldia), 6 origins (adds Maputo, Samarinda,
Novorossiysk), 54 routes, 30 vessels including Capesize, and 250 fixtures.
`commodity_prices.csv` and `freight_rates_history.csv` come out byte-identical
(same seed and date range), so forecasts do not change. Back up the CSVs first,
and update the row counts in this file afterwards.

## Known limitations (mention proactively if judges ask)
- Freight/commodity time series are synthetic — real day-to-day figures will differ, but distributional properties (range, volatility) are calibrated to real data.
- Port draft/DWT limits are approximate — actual berth-by-berth limits vary and change with dredging.
- Route distances are straight-line great-circle approximations, not actual sailed routes (which follow shipping lanes).
- Two sets of optimizer inputs exist: `sample_data.py` (3 hardcoded vessels/ports, the original demo) and `fleet_data.py` (the real `ports`/`vessels`/`routes` tables). The Monte Carlo risk simulation still runs on `sample_data`.
- `fleet_data.py`'s freight calibration is fitted on the **synthetic** `fixtures` table, so it recovers the generator's own pricing formula almost exactly (R2 = 1.0). On real fixture data the fit would be noisier — quote the method, not that R2.
- `draft_m`, `available_until` and `port_cost_usd` are synthetic estimates, not sourced figures.
- `fetch_real_data.py` now pulls **real** monthly commodity prices (World Bank Pink Sheet: Australian coal, US HRW wheat, maize) into `commodity_prices_real.csv` and **real** weekly US Gulf / Pacific NW ocean freight cost indices (USDA AMS, 2002-) into `freight_rates_real.csv`. When those files exist the forecasting modules use them instead of the synthetic CSVs, and routes `USG_OCEAN` / `PNW_OCEAN` are backed by real data. Everything else below is unchanged.
- The Baltic indices, the coal freight leg, the vessel fleet and the fixtures remain generated by `generate_data.py` from the real anchor values above. Live sources for a production version would be a Baltic Exchange subscription (freight indices), the World Bank "Pink Sheet" and USDA AMS reports (commodity prices and real ocean freight rates), and data.gov.in / port authority sites (Indian port data).
