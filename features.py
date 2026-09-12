"""
features.py — feature engineering shared logic for freight rate forecasting.

Per the team doc, Person 2 and Person 3 (commodity price forecasting) use a
similar toolset and should share feature-engineering utilities to avoid
duplicate work. This module is written generically enough (operating on any
`value` column) that Person 3 can reuse it for commodity prices too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MONSOON_MONTHS
from weather import CYCLONE_MONTHLY_WEIGHT


def add_lag_features(df: pd.DataFrame, col: str = "rate", lags=(1, 2, 4, 8, 12)) -> pd.DataFrame:
    """Adds lag_{n} columns for the given column."""
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    col: str = "rate",
    windows=(4, 8, 12),
) -> pd.DataFrame:
    """Adds rolling mean and rolling volatility (std) features.

    Windows are shifted by one row so they only use *past* values. Without
    the shift, row t's rolling mean includes the target value at t itself
    (data leakage): the model looks great in-sample but its residual-based
    confidence intervals come out far too narrow.
    """
    df = df.copy()
    past = df[col].shift(1)
    for w in windows:
        df[f"{col}_roll_mean_{w}"] = past.rolling(window=w, min_periods=1).mean()
        df[f"{col}_roll_std_{w}"] = past.rolling(window=w, min_periods=2).std()
    return df


def add_seasonality_flags(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Adds week-of-year (cyclical) and monsoon-season flag."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])

    week = dt.dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * week / 52)
    df["week_cos"] = np.cos(2 * np.pi * week / 52)

    df["month"] = dt.dt.month
    df["is_monsoon"] = df["month"].isin(MONSOON_MONTHS).astype(int)

    # Bay of Bengal cyclone-season weight (0-1, IMD climatology shape — see
    # weather.py). Route-level, not port-level: the freight indices (BDI/
    # BCI/BPI) reflect the whole east-coast India corridor, so we use the
    # basin-wide seasonal shape rather than any one port's exposure.
    df["cyclone_season_weight"] = df["month"].map(CYCLONE_MONTHLY_WEIGHT).fillna(0.05)
    return df


def add_bunker_features(df: pd.DataFrame, col: str = "bunker_price") -> pd.DataFrame:
    """Adds bunker price momentum features, if the column exists."""
    df = df.copy()
    if col not in df.columns:
        return df
    df[f"{col}_pct_change_4"] = df[col].pct_change(4)
    df[f"{col}_lag_1"] = df[col].shift(1)
    return df


def build_features(
    df: pd.DataFrame,
    target_col: str = "rate",
    date_col: str = "date",
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Full feature pipeline. Expects df with at least [date_col, target_col],
    optionally 'bunker_price'. Returns df with engineered columns.

    Shared by Person 2 (freight rates) and Person 3 (commodity prices) —
    just pass target_col='price' for commodity forecasting.
    """
    out = df.sort_values(date_col).reset_index(drop=True)
    out = add_lag_features(out, col=target_col)
    out = add_rolling_features(out, col=target_col)
    out = add_seasonality_flags(out, date_col=date_col)
    out = add_bunker_features(out)

    if dropna:
        out = out.dropna().reset_index(drop=True)

    return out


def feature_columns(df: pd.DataFrame, target_col: str = "rate", date_col: str = "date") -> list[str]:
    """Returns the list of engineered feature column names (excludes date/target)."""
    exclude = {date_col, target_col}
    return [c for c in df.columns if c not in exclude]
