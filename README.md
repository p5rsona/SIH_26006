# Freight Rate Forecasting

## Setup
```bash
pip install -r requirements.txt
```

## Files
- `config.py` — routes, DB connection settings
- `data_loader.py` — loads history from Person 1's MySQL DB; **falls back to a synthetic series automatically if the DB isn't reachable/seeded yet**, so you're not blocked
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
Nothing to change in your code — `data_loader.py` tries MySQL first automatically. Just set env vars (or edit `config.py`):
```bash
export FR_DB_HOST=... FR_DB_USER=... FR_DB_PASSWORD=... FR_DB_NAME=...
```

## Note on this sandbox
`statsmodels` and `xgboost` aren't installed in the environment this was built in (no network access to pip install), so `models.py`/`backtest.py`/`forecast.py` are reviewed carefully but not execution-tested here — `data_loader.py` and `features.py` (pandas/numpy only) were run and confirmed working. Run `pip install -r requirements.txt` then `python example_usage.py` on your machine to verify the SARIMAX/XGBoost paths before you present it to the team.
