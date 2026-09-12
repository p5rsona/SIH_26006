"""
PERSON 3 — Commodity Price Forecasting Pipeline
Matched to Person 1's real schema.sql / data_dictionary.md

Public contract (what Person 4 / 5 / the dashboard import):

    forecast_commodity_price(commodity, horizon_weeks) -> DataFrame[date, price, ci_lower, ci_upper]

Importing this module is cheap: data is loaded lazily on first use and the
full demo pipeline (XGBoost backtest + writing the forecast_<commodity>.csv
files) only runs when you execute the file directly:

    python person3_commodity_forecast.py
"""

from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
COMMODITY_CSV = ROOT / "commodity_prices.csv"
COMMODITIES = ["coal_newcastle", "wheat_gulf", "corn_gulf"]

# =====================================================================
# STEP 1A — Pull real data from Person 1's MySQL database
# =====================================================================
USE_REAL_DB = False   # <-- set True once Person 1's DB is loaded (credentials come from config.py / FR_DB_* env vars)


def load_from_mysql() -> pd.DataFrame:
    import mysql.connector
    from config import DB_CONFIG

    conn = mysql.connector.connect(**DB_CONFIG, connection_timeout=3)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT price_date AS date, commodity, price_usd_per_tonne AS price
            FROM commodity_prices
            ORDER BY commodity, price_date
            """
        )
        df = pd.DataFrame(cur.fetchall(), columns=["date", "commodity", "price"])
    finally:
        conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = df["price"].astype(float)
    return df


# =====================================================================
# STEP 1B — Fallback synthetic data
# =====================================================================
def load_synthetic_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-06-01", "2026-09-01", freq="B")

    rows = []
    commodities = {
        "coal_newcastle": 110,
        "wheat_gulf": 250,
        "corn_gulf": 190,
    }
    for name, base_price in commodities.items():
        n = len(dates)
        seasonal = 12 * np.sin(2 * np.pi * np.arange(n) / 252)
        trend = np.linspace(0, 20, n)
        noise = rng.normal(0, 6, n)
        price = base_price + seasonal + trend + noise
        for d, p in zip(dates, price):
            rows.append([d, name, round(max(p, 1), 2)])

    return pd.DataFrame(rows, columns=["date", "commodity", "price"])


def load_raw_data() -> pd.DataFrame:
    """MySQL (if USE_REAL_DB) -> commodity_prices.csv -> synthetic."""
    if USE_REAL_DB:
        try:
            return load_from_mysql()
        except Exception as e:
            print(f"MySQL load failed ({e}); falling back to CSV.")

    if COMMODITY_CSV.exists():
        df = pd.read_csv(COMMODITY_CSV)
        df = df.rename(columns={"price_date": "date", "price_usd_per_tonne": "price"})
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "commodity", "price"]]

    print("commodity_prices.csv not found; using synthetic data.")
    return load_synthetic_data()


# =====================================================================
# STEP 2 — Clean the data
# =====================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["date", "commodity"])
    df = df.sort_values(["commodity", "date"])
    df["price"] = df.groupby("commodity")["price"].transform(lambda x: x.interpolate())
    return df.reset_index(drop=True)


# =====================================================================
# STEP 3 — Feature engineering
# =====================================================================
def add_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    group["month"] = group["date"].dt.month
    group["day_of_week"] = group["date"].dt.dayofweek
    group["lag_1"] = group["price"].shift(1)
    group["lag_5"] = group["price"].shift(5)
    group["rolling_mean_5"] = group["price"].shift(1).rolling(5).mean()
    group["rolling_std_5"] = group["price"].shift(1).rolling(5).std()
    return group


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat(
        [add_features(group) for _, group in df.groupby("commodity")],
        ignore_index=True,
    )
    return out.dropna().reset_index(drop=True)


@lru_cache(maxsize=1)
def get_data() -> pd.DataFrame:
    """Loaded + cleaned + featurized data, cached for the whole process."""
    return build_feature_frame(clean_data(load_raw_data()))


# =====================================================================
# STEP 4 — Baseline model: SARIMAX
# =====================================================================
def sarimax_forecast(series: pd.Series, horizon_weeks: int):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Prices are business-daily. Give the index an explicit 'B' frequency
    # (filling any missing days) so the forecast comes back with real dates
    # instead of integer positions.
    series = series.asfreq("B").interpolate()

    horizon_days = horizon_weeks * 5
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5),
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fc = fit.get_forecast(steps=horizon_days)
    return fc.predicted_mean, fc.conf_int(alpha=0.05)


# =====================================================================
# STEP 5/6 — XGBoost model + backtest accuracy
# =====================================================================
FEATURES = ["month", "day_of_week", "lag_1", "lag_5", "rolling_mean_5", "rolling_std_5"]
TARGET = "price"


def xgboost_backtest(commodity: str = "coal_newcastle"):
    """Trains XGBoost on the first 85% of a commodity's history and scores
    one-step-ahead predictions on the last 15%. Returns (mape_pct, rmse)."""
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
    from xgboost import XGBRegressor

    df = get_data()
    sub = df[df.commodity == commodity].reset_index(drop=True)
    X, y = sub[FEATURES], sub[TARGET]

    split = int(len(sub) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    xgb_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, xgb_preds) * 100
    rmse = float(np.sqrt(mean_squared_error(y_test, xgb_preds)))
    return mape, rmse


# =====================================================================
# STEP 7 — Final function (this is what Person 4 & 5 use)
# =====================================================================
def forecast_commodity_price(commodity: str, horizon_weeks: int = 4) -> pd.DataFrame:
    df = get_data()
    sub = df[df.commodity == commodity].copy()
    if sub.empty:
        raise ValueError(f"No data for commodity '{commodity}'")

    series = sub.set_index("date")["price"]
    mean, ci = sarimax_forecast(series, horizon_weeks)

    return pd.DataFrame({
        "date": mean.index,
        "price": mean.values,
        "ci_lower": ci.iloc[:, 0].values,
        "ci_upper": ci.iloc[:, 1].values,
    }).reset_index(drop=True)


def main():
    print("STEP 1-3: Loading, cleaning and featurizing commodity data...")
    df = get_data()
    print(df.tail(3))
    print(f"{len(df)} rows, commodities: {df['commodity'].unique()}\n")

    print("STEP 4: SARIMAX 4-week forecast (coal_newcastle):")
    series = df[df.commodity == "coal_newcastle"].set_index("date")["price"]
    sarimax_mean, _ = sarimax_forecast(series, horizon_weeks=4)
    print(sarimax_mean.head(), "\n")

    print("STEP 5-6: XGBoost backtest (coal_newcastle)...")
    mape, rmse = xgboost_backtest("coal_newcastle")
    print(f"XGBoost — MAPE: {mape:.2f}%  RMSE: {rmse:.2f}\n")

    print("STEP 7: Writing forecast_<commodity>.csv files...")
    for commodity in COMMODITIES:
        out = forecast_commodity_price(commodity, horizon_weeks=4)
        print(f"--- {commodity} ---")
        print(out.head(), "\n")
        out.to_csv(ROOT / f"forecast_{commodity}.csv", index=False)

    print("Saved forecast_<commodity>.csv files.")


if __name__ == "__main__":
    main()
