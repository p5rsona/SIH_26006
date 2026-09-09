"""
example_usage.py — quick demo of the full pipeline. Run this to sanity
check the module end to end:

    python example_usage.py
"""

from backtest import compare_models
from data_loader import load_freight_rate_history
from forecast import forecast_freight_rate, forecast_with_disagreement

ROUTE = "C5"
HORIZON = 8

if __name__ == "__main__":
    print(f"=== Loading history for {ROUTE} ===")
    history = load_freight_rate_history(ROUTE)
    print(f"{len(history)} weeks loaded, {history['date'].min().date()} -> {history['date'].max().date()}\n")

    print(f"=== Backtest: SARIMAX vs XGBoost ({HORIZON}-week horizon) ===")
    summary = compare_models(history, horizon_weeks=HORIZON, n_splits=4)
    print(summary.to_string(index=False), "\n")

    def _fmt(df):
        # .round(1) can't touch the datetime column directly (pandas 2+
        # warns and no-ops on it) — round only the numeric columns instead
        out = df.copy()
        numeric_cols = out.select_dtypes(include="number").columns
        out[numeric_cols] = out[numeric_cols].round(1)
        out["date"] = out["date"].dt.date
        return out

    print(f"=== Forecast: {ROUTE}, next {HORIZON} weeks (SARIMAX) ===")
    fc = forecast_freight_rate(ROUTE, horizon_weeks=HORIZON, model="sarimax")
    print(_fmt(fc).to_string(index=False), "\n")

    print(f"=== Forecast: {ROUTE}, next {HORIZON} weeks (XGBoost) ===")
    fc_xgb = forecast_freight_rate(ROUTE, horizon_weeks=HORIZON, model="xgboost")
    print(_fmt(fc_xgb).to_string(index=False), "\n")

    print(f"=== Forecast with model disagreement (for Person 5's Monte Carlo layer) ===")
    fc_disagree = forecast_with_disagreement(ROUTE, horizon_weeks=HORIZON)
    print(_fmt(fc_disagree).to_string(index=False))
