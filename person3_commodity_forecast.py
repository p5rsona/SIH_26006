"""
PERSON 3 — Commodity Price Forecasting Pipeline
Matched to Person 1's real schema.sql / data_dictionary.md
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# STEP 1A — Pull real data from Person 1's MySQL database
# =====================================================================
USE_REAL_DB = False   # <-- change to True once you fill in credentials below

def load_from_mysql():
    from sqlalchemy import create_engine

    DB_USER = "your_username"
    DB_PASS = "your_password"
    DB_HOST = "localhost"
    DB_NAME = "sih_shipping"

    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

    query = """
        SELECT price_date AS date, commodity, price_usd_per_tonne AS price
        FROM commodity_prices
        ORDER BY commodity, price_date
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    return df


# =====================================================================
# STEP 1B — Fallback synthetic data
# =====================================================================
def load_synthetic_data():
    np.random.seed(42)
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
        noise = np.random.normal(0, 6, n)
        price = base_price + seasonal + trend + noise
        for d, p in zip(dates, price):
            rows.append([d, name, round(max(p, 1), 2)])

    return pd.DataFrame(rows, columns=["date", "commodity", "price"])


print("STEP 1: Loading Person 1's commodity data...")

df = pd.read_csv("commodity_prices.csv")

df = df.rename(columns={
    "price_date": "date",
    "price_usd_per_tonne": "price"
})

df["date"] = pd.to_datetime(df["date"])

print(df.head())
print(f"Loaded {len(df)} rows, commodities: {df['commodity'].unique()}\n")

# =====================================================================
# STEP 2 — Clean the data
# =====================================================================
print("STEP 2: Cleaning...")
df = df.drop_duplicates(subset=["date", "commodity"])
df = df.sort_values(["commodity", "date"])
df["price"] = df.groupby("commodity")["price"].transform(lambda x: x.interpolate())
print("Missing values after cleaning:", df["price"].isna().sum(), "\n")


# =====================================================================
# STEP 3 — Feature engineering
# =====================================================================
print("STEP 3: Feature engineering...")

def add_features(group):
    group = group.copy()
    group["month"] = group["date"].dt.month
    group["day_of_week"] = group["date"].dt.dayofweek
    group["lag_1"] = group["price"].shift(1)
    group["lag_5"] = group["price"].shift(5)
    group["rolling_mean_5"] = group["price"].shift(1).rolling(5).mean()
    group["rolling_std_5"] = group["price"].shift(1).rolling(5).std()
    return group

df = pd.concat(
    [add_features(group) for _, group in df.groupby("commodity")],
    ignore_index=True
)
df = df.dropna().reset_index(drop=True)
print(df.tail(3), "\n")


# =====================================================================
# STEP 4 — Baseline model: SARIMAX
# =====================================================================
print("STEP 4: SARIMAX baseline...")
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarimax_forecast(series, horizon_weeks):
    horizon_days = horizon_weeks * 5
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5),
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fc = fit.get_forecast(steps=horizon_days)
    return fc.predicted_mean, fc.conf_int(alpha=0.05)

example = df[df.commodity == "coal_newcastle"]
series = example.set_index("date")["price"]
sarimax_mean, sarimax_ci = sarimax_forecast(series, horizon_weeks=4)
print("SARIMAX 4-week forecast (coal_newcastle):")
print(sarimax_mean.head(), "\n")


# =====================================================================
# STEP 5 — Stronger model: XGBoost
# =====================================================================
print("STEP 5: XGBoost model...")
from xgboost import XGBRegressor

FEATURES = ["month", "day_of_week", "lag_1", "lag_5", "rolling_mean_5", "rolling_std_5"]
TARGET = "price"

sub = df[df.commodity == "coal_newcastle"].reset_index(drop=True)
X, y = sub[FEATURES], sub[TARGET]

split = int(len(sub) * 0.85)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

xgb_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
print("XGBoost sample predictions:", xgb_preds[:5], "\n")


# =====================================================================
# STEP 6 — Backtest accuracy
# =====================================================================
print("STEP 6: Backtest accuracy...")
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

mape = mean_absolute_percentage_error(y_test, xgb_preds) * 100
rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
print(f"XGBoost — MAPE: {mape:.2f}%  RMSE: {rmse:.2f}\n")


# =====================================================================
# STEP 7 — Final function (this is what Person 4 & 5 use)
# =====================================================================
print("STEP 7: Packaging forecast_commodity_price()...")

def forecast_commodity_price(commodity: str, horizon_weeks: int = 4) -> pd.DataFrame:
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

for commodity in ["coal_newcastle", "wheat_gulf", "corn_gulf"]:
    out = forecast_commodity_price(commodity, horizon_weeks=4)
    print(f"--- {commodity} ---")
    print(out.head(), "\n")
    out.to_csv(f"forecast_{commodity}.csv", index=False)

print("Saved forecast_<commodity>.csv files.")