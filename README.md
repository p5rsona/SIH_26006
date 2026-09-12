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

## Running the whole project
```bash
pip install -r requirements.txt        # includes ortools, streamlit, plotly
python example_usage.py                # freight forecasting + backtest
python person3_commodity_forecast.py   # rewrites forecast_<commodity>.csv
python optimizer.py                    # chartering plan
python person5_risk_simulation.py      # Monte Carlo savings
streamlit run dashboard.py
```
Load the DB with `python load_to_mysql.py --user root` (password via `--password`, the `FR_DB_PASSWORD` env var, or a prompt).
