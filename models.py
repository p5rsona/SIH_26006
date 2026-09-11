"""
models.py — forecasting models for freight rates.

Two models, common interface (fit / predict):
  - SarimaxModel: classical time-series baseline, gives native confidence
    intervals.
  - XgbForecaster: gradient-boosted trees on engineered features
    (lags, rolling vol, bunker price, seasonality) with recursive
    multi-step forecasting and residual-based confidence intervals.

Judges want to see model validation, not just predictions — see
backtest.py for the MAPE/RMSE comparison between these two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features import add_bunker_features, add_lag_features, add_rolling_features, add_seasonality_flags, feature_columns

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


@dataclass
class ForecastResult:
    dates: pd.DatetimeIndex
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


class SarimaxModel:
    """
    SARIMAX baseline for weekly data with annual seasonality.

    Default order is (1,1,1)x(0,1,1,52) — an "airline model" style
    seasonal ARIMA. This is deliberately lighter than a full
    (1,1,1)x(1,1,1,52): with ~5 years of weekly data (~260 points),
    double seasonal differencing (D=1 at lag 52) already eats ~52+
    observations, and adding a seasonal AR term on top of that gave the
    optimizer too little signal to converge reliably — it landed in a
    non-stationary region and produced exploding forecasts (values like
    1e106). Keeping enforce_stationarity/enforce_invertibility ON (the
    statsmodels default) prevents that failure mode by construction.

    If the primary order still fails to fit cleanly, .fit() automatically
    retries with progressively simpler fallback orders rather than
    silently returning a broken/explosive model — important since this
    runs inside a walk-forward backtest with many folds, any one of
    which could hit a bad optimization.
    """

    # (order, seasonal_order) tried in sequence until one fits cleanly
    _FALLBACK_SPECS = [
        ((1, 1, 1), (0, 1, 1, 52)),   # airline-style seasonal model (default)
        ((1, 1, 0), (0, 1, 0, 52)),   # simpler seasonal, no seasonal MA
        ((1, 1, 1), (0, 0, 0, 0)),    # non-seasonal fallback (last resort)
    ]

    def __init__(self, order=None, seasonal_order=None):
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels is not installed. `pip install statsmodels`.")
        # explicit order/seasonal_order (if given) is tried first, then
        # the built-in fallback ladder
        self._specs = (
            [(order, seasonal_order)] if order and seasonal_order else []
        ) + self._FALLBACK_SPECS
        self._fitted = None
        self._last_date = None
        self._freq = None
        self.order = None            # set to whichever spec actually succeeded
        self.seasonal_order = None

    def fit(self, df: pd.DataFrame, target_col: str = "rate", date_col: str = "date"):
        # resample (not asfreq): asfreq only keeps rows that fall exactly on
        # a Monday, so daily or non-Monday data would become all-NaN
        series = (
            df.assign(**{date_col: pd.to_datetime(df[date_col])})
            .set_index(date_col)[target_col]
            .resample("W-MON").mean()
            .interpolate()  # SARIMAX needs no gaps
            .dropna()
        )

        # historical range, used to sanity-check the fit isn't explosive
        hist_min, hist_max = series.min(), series.max()
        hist_span = max(hist_max - hist_min, 1.0)

        # A seasonal model needs at least two full seasons of data; with less
        # (e.g. Person 1's ~18 months of history) skip straight to the
        # non-seasonal specs instead of forcing a fit that can't converge.
        specs = [
            (o, so) for o, so in self._specs
            if not so[3] or len(series) >= 2 * so[3]
        ] or [self._FALLBACK_SPECS[-1]]

        last_error = None
        for order, seasonal_order in specs:
            try:
                model = SARIMAX(
                    series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
                fitted = model.fit(disp=False, maxiter=200)

                # sanity check: a one-step-ahead forecast should be in a
                # plausible range relative to history, not 1e30+
                probe = fitted.get_forecast(steps=1).predicted_mean.iloc[0]
                if not np.isfinite(probe) or abs(probe - hist_max) > 50 * hist_span:
                    raise ValueError(
                        f"Fit produced an implausible forecast ({probe:.3g}); "
                        f"treating as non-convergent."
                    )

                self._fitted = fitted
                self.order, self.seasonal_order = order, seasonal_order
                break

            except Exception as e:
                last_error = e
                continue
        else:
            raise RuntimeError(
                f"SARIMAX failed to fit with all fallback orders. Last error: {last_error}"
            )

        self._last_date = series.index[-1]
        self._freq = series.index.freq
        return self

    def predict(self, horizon_weeks: int, alpha: float = 0.05) -> ForecastResult:
        if self._fitted is None:
            raise RuntimeError("Call .fit() before .predict()")

        fc = self._fitted.get_forecast(steps=horizon_weeks)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=alpha)

        dates = pd.date_range(
            start=self._last_date + self._freq, periods=horizon_weeks, freq=self._freq
        )

        return ForecastResult(
            dates=dates,
            point=mean.values,
            lower=ci.iloc[:, 0].values,
            upper=ci.iloc[:, 1].values,
        )


class XgbForecaster:
    """
    Gradient-boosted trees on engineered features, with recursive multi-step
    forecasting: predict week t+1, feed it back in to build features for
    t+2, and so on. Confidence interval is approximated from backtest
    residual std (Gaussian assumption) since XGBoost has no native CI.
    """

    def __init__(self, **xgb_params):
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed. `pip install xgboost`.")
        default_params = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        default_params.update(xgb_params)
        self.model = XGBRegressor(**default_params)
        self.target_col = "rate"
        self.date_col = "date"
        self._residual_std = None
        self._history = None  # raw (non-featurized) history, needed for recursive forecasting

    def fit(self, df: pd.DataFrame, target_col: str = "rate", date_col: str = "date"):
        from features import build_features  # local import avoids circularity concerns

        self.target_col = target_col
        self.date_col = date_col
        self._history = df.sort_values(date_col).reset_index(drop=True).copy()

        feat_df = build_features(df, target_col=target_col, date_col=date_col)
        cols = feature_columns(feat_df, target_col=target_col, date_col=date_col)
        self._feature_cols = cols

        X, y = feat_df[cols], feat_df[target_col]

        # CI width comes from one-step-ahead errors on a held-out tail (last
        # 20%), not in-sample residuals — boosted trees nearly memorize the
        # training set, so in-sample residuals make the CI far too narrow.
        n_hold = max(int(len(X) * 0.2), 1)
        if len(X) - n_hold >= 20:
            self.model.fit(X.iloc[:-n_hold], y.iloc[:-n_hold])
            hold_err = y.iloc[-n_hold:].values - self.model.predict(X.iloc[-n_hold:])
            self._residual_std = float(np.sqrt(np.mean(hold_err ** 2)))
        else:
            self._residual_std = None

        self.model.fit(X, y)  # final model uses all the data
        if self._residual_std is None:
            self._residual_std = float(np.std(y.values - self.model.predict(X)))
        return self

    def predict(self, horizon_weeks: int, alpha: float = 0.05) -> ForecastResult:
        if self._history is None:
            raise RuntimeError("Call .fit() before .predict()")

        from features import build_features
        from scipy.stats import norm

        z = norm.ppf(1 - alpha / 2)
        history = self._history.copy()
        history[self.date_col] = pd.to_datetime(history[self.date_col])
        last_date = history[self.date_col].max()
        freq = pd.Timedelta(weeks=1)

        preds = []
        for step in range(horizon_weeks):
            next_date = last_date + freq * (step + 1)

            # Append the week being forecast with an unknown target, then build
            # its features: lags/rolling windows come from the weeks before it
            # (actual history plus earlier predictions). Previously the model
            # was fed the features of the *last observed* week, so it predicted
            # that week again and labelled it as next week (off by one).
            new_row = {self.date_col: next_date, self.target_col: np.nan}
            if "bunker_price" in history.columns:
                new_row["bunker_price"] = history["bunker_price"].iloc[-1]  # carry forward last known
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

            feat_df = build_features(history, target_col=self.target_col, date_col=self.date_col, dropna=False)
            row = feat_df.iloc[[-1]][self._feature_cols].fillna(0)

            yhat = float(self.model.predict(row)[0])
            preds.append(yhat)

            # record the prediction so the next step's lag features can use it
            history.loc[history.index[-1], self.target_col] = yhat

        preds = np.array(preds)
        # widen the interval with forecast horizon (uncertainty compounds)
        horizon_factor = np.sqrt(np.arange(1, horizon_weeks + 1))
        margin = z * self._residual_std * horizon_factor

        dates = pd.date_range(start=last_date + freq, periods=horizon_weeks, freq="W-MON")
        return ForecastResult(
            dates=dates,
            point=preds,
            lower=preds - margin,
            upper=preds + margin,
        )
