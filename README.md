# Freight Rate Forecasting (Person 2)

## Setup
```bash
pip install -r requirements.txt
```

## Files
- `config.py` — routes, CSV fallback paths
- `db.py` — the Supabase (Postgres) connection, shared by every reader
- `data_loader.py` — loads history from Supabase, then `freight_rates_history.csv`, then a synthetic series. Each route maps to a Baltic index (`config.ROUTE_TO_INDEX`: C5/C3 → BCI, P1A_82 → BPI; TD3C has no index in the data so it stays synthetic), averaged from daily to weekly
- `weather.py` — Bay of Bengal weather & cyclone risk: a 0-1 risk score per (port, date window) that feeds a `cyclone_season_weight` regressor into the forecasts, an expected weather-delay $ cost + hard block into the optimizer, and a severity multiplier into the Monte Carlo. Run `python weather.py` to print the climatology table
- `features.py` — lag/rolling-volatility/seasonality/monsoon/cyclone-season/bunker-price features (shared design so Person 3 can reuse for commodity prices — just pass a different `target_col`)
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

## Once the Supabase DB is seeded
Nothing to change in your code — `data_loader.py` tries Supabase first automatically.
Just put the connection URL in `.env` (copy `.env.example`):
```bash
SUPABASE_DB_URL=postgresql://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres
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
| `GET /weather/risk?port=Paradip&laycan_start=2026-10-20&laycan_end=2026-11-05` | — | risk score, expected delay days, weather-risk cost, hard-blocked flag |
| `GET /weather/climatology` | — | monthly cyclone-risk score per port (for charting) |

The Pydantic models in `api.py` are the team's interface contracts written as code:
if a module returns a different shape, the API fails there instead of silently
breaking the dashboard.

The dashboard uses the API when it is running and falls back to importing the
modules in-process when it is not, so nothing breaks if you forget to start it.
Each tab's caption says which path it used. Override the address with the
`API_URL` environment variable.

## Database: Supabase (`db.py`)
Everything that reads the database goes through `db.py`, so there is one place to
configure and one connection URL. No database at all is fine too — every reader
falls back to the CSVs.

```bash
cp .env.example .env        # then paste your SUPABASE_DB_URL into it
pip install sqlalchemy psycopg2-binary python-dotenv
python load_to_db.py        # creates the tables from schema_postgres.sql + loads the seed CSVs
python db.py                # prints whether Supabase is reachable, and the vessel count
```
`.env` is git-ignored — keep the password out of the repo. `GET /health` and the
dashboard sidebar report the same connection status as `python db.py`.

Get the URL from the Supabase dashboard -> Project Settings -> Database ->
Connection string. **The direct host (`db.<ref>.supabase.co:5432`) is IPv6-only**,
so on an IPv4-only network use the *Connection pooling* URI instead — username
`postgres.<project-ref>`, port 6543. `db.py` says which problem it hit when it
can't connect.

`schema_postgres.sql` is the authoritative schema. `schema.sql` (MySQL) and
`load_to_mysql.py` are left over from the old local-MySQL setup and are no longer
used.

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
Load the DB with `python load_to_db.py` (connection URL from `.env`).
