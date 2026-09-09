# Person 3 — Commodity Price Forecasting

## Objective
Forecast future FOB commodity prices for the maritime procurement
and vessel chartering optimization system.

## Commodities
- Newcastle coal
- US Gulf wheat
- US Gulf corn

## Input
The model uses the commodity_prices.csv dataset provided by the
project data pipeline.

## Data Processing
The pipeline:
1. Loads commodity price data.
2. Removes duplicate records.
3. Sorts the data by commodity and date.
4. Handles missing prices using interpolation.
5. Creates lag and rolling statistical features.

## Forecasting Models
Two forecasting approaches are used:
- SARIMAX baseline model
- XGBoost regression model

## Evaluation
The XGBoost model is evaluated using:
- MAPE
- RMSE

## Output
The system generates:
- forecast_coal_newcastle.csv
- forecast_wheat_gulf.csv
- forecast_corn_gulf.csv

Each forecast contains:
- date
- predicted price
- confidence interval lower bound
- confidence interval upper bound

## Main Function

forecast_commodity_price(commodity, horizon_weeks)

This function is designed to provide commodity price forecasts for
the other components of the project.