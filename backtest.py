"""
backtest.py — out-of-sample validation (MAPE / RMSE) for the freight rate
models. Judges want to see model validation, not just predictions, so this
is meant to produce a small report/table you can drop straight into slides.

Uses a rolling-origin (walk-forward) backtest: for each of `n_splits` folds,
train on everything up to a cutoff, forecast `horizon_weeks` ahead, and
score against the actual held-out values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models import NaiveModel, SarimaxModel, XgbForecaster


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _rolling_splits(n_rows: int, horizon: int, n_splits: int, min_train: int):
    """Yields (train_end_idx, test_end_idx) for walk-forward folds."""
    max_start = n_rows - horizon
    if max_start <= min_train:
        raise ValueError("Not enough data for the requested horizon/min_train.")

    fold_starts = np.linspace(min_train, max_start, num=n_splits, dtype=int)
    for start in fold_starts:
        yield start, start + horizon


def backtest_model(
    model_cls,
    df: pd.DataFrame,
    target_col: str = "rate",
    date_col: str = "date",
    horizon_weeks: int = 8,
    n_splits: int = 5,
    min_train: int = 60,
    **model_kwargs,
) -> pd.DataFrame:
    """
    Runs a walk-forward backtest for a given model class (SarimaxModel or
    XgbForecaster). Returns a DataFrame with one row per fold: [fold, mape, rmse].
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    rows = []

    for fold, (train_end, test_end) in enumerate(
        _rolling_splits(len(df), horizon_weeks, n_splits, min_train), start=1
    ):
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        if len(test_df) < horizon_weeks:
            continue  # incomplete fold at the tail, skip

        model = model_cls(**model_kwargs)
        model.fit(train_df, target_col=target_col, date_col=date_col)
        result = model.predict(horizon_weeks)

        y_true = test_df[target_col].values
        y_pred = result.point[: len(y_true)]

        rows.append({
            "fold": fold,
            "train_size": len(train_df),
            "mape": mape(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
        })

    return pd.DataFrame(rows)


def compare_models(
    df: pd.DataFrame,
    target_col: str = "rate",
    date_col: str = "date",
    horizon_weeks: int = 8,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Runs the backtest for the naive benchmark, SARIMAX and XGBoost, and
    returns a summary table (mean MAPE / RMSE per model) — this is the table
    to put in the slide deck.

    Read it against the naive row, not in isolation. On freight rates at a
    multi-week horizon, persistence is hard to beat; if SARIMAX only matches
    it on MAPE, the argument for SARIMAX is its calibrated confidence
    interval — which the Monte Carlo layer needs and a point forecast cannot
    provide — not point accuracy.
    """
    summaries = []

    # Naive first: it is the benchmark the other two have to justify
    # themselves against, and reading the table top-down should make that
    # obvious. A model that does not beat persistence is not adding accuracy,
    # whatever else it adds.
    for name, cls in [
        ("Naive (random walk)", NaiveModel),
        ("SARIMAX", SarimaxModel),
        ("XGBoost", XgbForecaster),
    ]:
        try:
            fold_results = backtest_model(
                cls, df, target_col=target_col, date_col=date_col,
                horizon_weeks=horizon_weeks, n_splits=n_splits,
            )
            summaries.append({
                "model": name,
                "mean_mape": fold_results["mape"].mean(),
                "mean_rmse": fold_results["rmse"].mean(),
                "n_folds": len(fold_results),
            })
        except Exception as e:
            summaries.append({
                "model": name,
                "mean_mape": None,
                "mean_rmse": None,
                "n_folds": 0,
                "error": str(e),
            })

    return pd.DataFrame(summaries)


if __name__ == "__main__":
    from data_loader import load_freight_rate_history

    df = load_freight_rate_history("C5")
    print("Backtesting SARIMAX vs XGBoost on route C5...\n")
    summary = compare_models(df, horizon_weeks=8, n_splits=4)
    print(summary.to_string(index=False))
