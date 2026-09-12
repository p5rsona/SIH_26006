# Freight Rate Forecasting (Person 2)

## Setup
```bash
pip install -r requirements.txt
```

## Files
- `config.py` — routes, DB connection settings
- `data_loader.py` — loads history from Person 1's MySQL DB, then `freight_rates_history.csv`, then a synthetic series. Each route maps to a Baltic index (`config.ROUTE_TO_INDEX`: C5/C3 → BCI, P1A_82 → BPI; TD3C has no index in the data so it stays synthetic), averaged from daily to weekly
- `features.py` — lag/rolling-volatility/seasonality/monsoon/bunker-price features (shared design so Person 3 can reuse for commodity prices — just pass a different `target_col`)
- `models.py` — `SarimaxModel` and `XgbForecaster`, common `fit()`/`predict()` interface
- `backtest.py` — walk-forward MAPE/RMSE backtest, `compare_models()` gives a summary table for slides
- `forecast.py` — **the output contract**: `forecast_freight_rate(route, horizon_weeks) -> DataFrame[date, rate, ci_lower, ci_upper]`
- `example_usage.py` — run this to see the whole pipeline end to end

## Quick start
```bash
python example_usage.py
```

## For Person 4 / Person 5
```python
from forecast import forecast_freight_rate

df = forecast_freight_rate("C5", horizon_weeks=8)
# columns: date, rate, ci_lower, ci_upper
```

### For Person 5 specifically — model disagreement as a risk signal
```python
from forecast import forecast_with_disagreement

df = forecast_with_disagreement("C5", horizon_weeks=8)
# columns: date, rate, ci_lower, ci_upper, xgb_rate, disagreement, disagreement_pct
```
SARIMAX and XGBoost make very different structural assumptions, so when
their point forecasts diverge, that gap is itself a useful signal — it
means there's more genuine uncertainty in the market than SARIMAX's own
CI (which only reflects uncertainty *within* its own assumptions) would
suggest on its own. Fold `disagreement_pct` into the Monte Carlo scenario
variance as an extra source of spread, on top of the residual-based
variance you're already sampling from SARIMAX's CI.

## Once Person 1's DB is live
Nothing to change in your code — `data_loader.py` tries MySQL (`sih_shipping`) first automatically. Just set env vars (or edit `config.py`):
```bash
export FR_DB_HOST=... FR_DB_USER=... FR_DB_PASSWORD=... FR_DB_NAME=...
```

## API (backend service)
`api.py` puts a FastAPI service in front of the whole pipeline, so the dashboard
(or any other client — an ERP, Postman, curl) talks HTTP instead of importing
Python modules.

```bash
python -m uvicorn api:app --reload --port 8000
```
Then open **http://localhost:8000/docs** for interactive documentation where every
endpoint can be tried from the browser.

| Endpoint | Owner | Returns |
|---|---|---|
| `GET /health` | — | service status + which modules imported |
| `GET /meta` | — | valid routes / commodities / models |
| `GET /forecast/freight?route=C5&weeks=8` | Person 2 | date, rate, ci_lower, ci_upper |
| `GET /forecast/commodity?commodity=coal_newcastle&weeks=4` | Person 3 | date, price, ci_lower, ci_upper |
| `POST /optimize` | Person 4 | plan (cargo, vessel, port, qty, cost_breakdown) + total_cost |
| `POST /simulate` | Person 5 | scenario costs + summary |

The Pydantic models in `api.py` are the team's interface contracts written as code:
if a module returns a different shape, the API fails there instead of silently
breaking the dashboard.

The dashboard uses the API when it is running and falls back to importing the
modules in-process when it is not, so nothing breaks if you forget to start it.
Each tab's caption says which path it used. Override the address with the
`API_URL` environment variable.

## Database: MySQL or Supabase (`db.py`)
Everything that reads the database goes through `db.py`, so the backend is one
environment variable. No database at all is fine too — every reader falls back to
the CSVs.

```bash
# A) local MySQL (default)
set DB_BACKEND=mysql
set FR_DB_PASSWORD=yourpassword
python load_to_db.py            # or the original load_to_mysql.py

# B) Supabase (Project Settings -> Database -> Connection string -> URI)
set DB_BACKEND=postgres
set SUPABASE_DB_URL=postgresql://postgres.<ref>:<password>@<host>:6543/postgres
python load_to_db.py
```
`python db.py` prints which backend is active and whether it is reachable; so does
`GET /health` and the dashboard sidebar. Copy `.env.example` for the full list of
settings. `schema_postgres.sql` is the Postgres translation of `schema.sql` —
**keep the two in sync** when you change a table.

Why Supabase: one shared database for the whole team instead of six local MySQL
installs, and the dashboard runs from any machine. Watch out for two things on the
free plan: 500 MB (plenty here) and **projects pause after a week of inactivity**,
so touch it a day or two before the demo.

## Real market data (`fetch_real_data.py`)
Replaces two of the synthetic series with real published data — no API key needed:

```bash
python fetch_real_data.py --dry-run   # download + report, write nothing
python fetch_real_data.py             # write commodity_prices_real.csv + freight_rates_real.csv
```

| Source | What it gives | Frequency |
|---|---|---|
| World Bank "Pink Sheet" | Australian coal, US wheat (HRW), maize — FOB benchmark prices | monthly, 1960- |
| USDA AMS grain transport cost indicators | US Gulf + Pacific NW ocean vessel rate indices | weekly, 2002- |

Once the files exist they are used automatically: `person3_commodity_forecast.py`
prefers `commodity_prices_real.csv` (and switches its SARIMAX spec to monthly),
and `data_loader.py` serves two new real routes — **USG_OCEAN** and **PNW_OCEAN** —
from `freight_rates_real.csv`. Delete the files to go back to synthetic data.

Still synthetic, because no free source exists: the Baltic BDI/BCI/BPI/BSI indices
(subscription; Trading Economics' free tier is the planned replacement), coal
freight, the vessel fleet and the fixtures.

## Real fleet (`fleet_data.py`)
`sample_data.py` gives the optimizer 3 hardcoded vessels. `fleet_data.py` builds the
same inputs from Person 1's real tables instead:

```python
from fleet_data import get_fleet_inputs
from optimizer import optimize

cargo_list, vessel_list, constraints = get_fleet_inputs(laycan_start="2026-08-24", laycan_end="2026-09-15")
plan = optimize(cargo_list, vessel_list, constraints)
```
- **Vessels**: every ship in `vessels` open during the laycan window (draft, DWT,
  availability from the table).
- **Freight $/tonne**: calibrated from the historical `fixtures` table — a linear fit of
  realised $/tonne on (index level, route distance) — evaluated at **Person 2's SARIMAX
  forecast** of the relevant Baltic index (BCI/BPI/BSI by vessel class). This is how the
  freight forecast finally reaches the optimizer.
- **Voyage fixed cost**: the ballast (empty) leg from where the vessel is open to the load
  port — days x `consumption_tpd_ballast` x bunker price — so a ship already near the load
  port can beat a cheaper $/tonne rate.
- **Port cost**: `ports.port_cost_usd`.

Because freight and ballast cost vary by cargo/vessel/port, they reach the optimizer as
`constraints["cost_matrix"]`. Without that key `optimize()` behaves exactly as before.

In the dashboard, the Optimizer tab has a **Fleet** switch (real fleet vs sample). Over
HTTP: `POST /optimize {"use_real_fleet": true}`, and `GET /fleet` lists the open vessels.

## Running the whole project
```bash
pip install -r requirements.txt        # includes ortools, streamlit, plotly
python example_usage.py                # freight forecasting + backtest
python person3_commodity_forecast.py   # rewrites forecast_<commodity>.csv
python optimizer.py                    # chartering plan
python person5_risk_simulation.py      # Monte Carlo savings
python -m uvicorn api:app --port 8000   # backend API (optional)
streamlit run dashboard.py             # dashboard
```
On Windows, `run_all.bat` starts the API and the dashboard together
(`run_all.sh` on Linux/macOS).
Load the DB with `python load_to_mysql.py --user root` (password via `--password`, the `FR_DB_PASSWORD` env var, or a prompt).
