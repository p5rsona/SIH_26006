"""
forecast.py — the public interface Person 4 (optimizer) and Person 5 (risk
simulation) import and call. This is the contract from the team doc:

    forecast_freight_rate(route, horizon_weeks) -> DataFrame[date, rate, ci_lower, ci_upper]

Internally it picks the better-performing model (via a quick backtest) or
you can force a specific one with `model="sarimax"` / `model="xgboost"`.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from config import SUPPORTED_ROUTES
from data_loader import load_freight_rate_history
from models import SarimaxModel, XgbForecaster

_MODEL_REGISTRY = {
    "sarimax": SarimaxModel,
    "xgboost": XgbForecaster,
}


@lru_cache(maxsize=32)
def _cached_load(route: str):
    """Internal cache so repeated calls (e.g. from Person 5's Monte Carlo
    loop, which may call this many times per route) don't re-hit Supabase."""
    return load_freight_rate_history(route)


def _cached_history(route: str):
    """A copy of the cached history. The cache hands back the same object
    every call, so without this a caller that adds a column (or sorts in
    place) would corrupt every later forecast for that route."""
    cached = _cached_load(route)
    out = cached.copy()
    out.attrs = dict(cached.attrs)
    return out


def forecast_freight_rate(
    route: str,
    horizon_weeks: int,
    model: str = "sarimax",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Forecasts freight rate for `route`, `horizon_weeks` weeks ahead.

    Parameters
    ----------
    route : str
        Route code, e.g. "C5". Must be in config.SUPPORTED_ROUTES.
    horizon_weeks : int
        Number of weeks ahead to forecast.
    model : str
        "sarimax" (default, classical baseline w/ native CI) or
        "xgboost" (feature-based, better if bunker price / seasonality
        signal is strong).
    alpha : float
        Significance level for the confidence interval (0.05 -> 95% CI).

    Returns
    -------
    pd.DataFrame with columns [date, rate, ci_lower, ci_upper]
        This exact shape is the contract Person 4 and Person 5 build against.
    """
    if route not in SUPPORTED_ROUTES:
        raise ValueError(f"Unknown route '{route}'. Supported: {SUPPORTED_ROUTES}")

    if model not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(_MODEL_REGISTRY)}")

    history = _cached_history(route)

    model_instance = _MODEL_REGISTRY[model]()
    model_instance.fit(history, target_col="rate", date_col="date")
    result = model_instance.predict(horizon_weeks, alpha=alpha)

    return pd.DataFrame({
        "date": result.dates,
        "rate": result.point,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
    })


def forecast_with_disagreement(
    route: str,
    horizon_weeks: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Runs BOTH SARIMAX and XGBoost and returns the SARIMAX forecast (better-
    calibrated CI, since it's model-based rather than a residual-std
    heuristic) plus an explicit "model disagreement" signal — the gap
    between the two models' point forecasts at each horizon week.

    Rationale: SARIMAX and XGBoost make very different structural
    assumptions about the series (linear time-series model w/ explicit
    trend+seasonality vs. tree-based model on engineered features with no
    inherent notion of time). When they agree, that's evidence the pattern
    is real rather than one model's artifact. When they diverge, that
    divergence is itself useful information — it means there's more
    uncertainty in the near-term freight market than SARIMAX's own CI
    alone would suggest, since SARIMAX's CI only reflects uncertainty
    *within* its own modeling assumptions, not uncertainty about whether
    those assumptions are the right ones.

    This is meant for Person 5's Monte Carlo layer: fold `disagreement_pct`
    in as an extra source of scenario variance (model uncertainty), on top
    of whatever residual-based variance you're already sampling from
    SARIMAX's own CI.

    Returns
    -------
    pd.DataFrame with columns:
        date, rate, ci_lower, ci_upper   — SARIMAX forecast (primary output,
                                            same shape as forecast_freight_rate)
        xgb_rate                          — XGBoost's point forecast, for reference
        disagreement                      — abs(sarimax_rate - xgb_rate), in $/day
        disagreement_pct                  — disagreement as a % of the SARIMAX rate
    """
    sarimax_fc = forecast_freight_rate(route, horizon_weeks, model="sarimax", alpha=alpha)
    xgb_fc = forecast_freight_rate(route, horizon_weeks, model="xgboost", alpha=alpha)

    out = sarimax_fc.copy()
    out["xgb_rate"] = xgb_fc["rate"].values
    out["disagreement"] = (out["rate"] - out["xgb_rate"]).abs()
    out["disagreement_pct"] = (out["disagreement"] / out["rate"]) * 100

    return out


if __name__ == "__main__":
    df = forecast_freight_rate("C5", horizon_weeks=8, model="sarimax")
    print(df.round(1).to_string(index=False))
