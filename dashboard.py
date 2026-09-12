"""
dashboard.py
============
Person 6 — Streamlit Dashboard & Integration Lead

Run with:
    streamlit run dashboard.py

All dashboard data frames follow the team contracts:
- forecast: date, rate, ci_lower, ci_upper
- optimizer: cargo, vessel, port, qty, cost_breakdown
- risk: scenario_id, optimized_cost, baseline_cost, savings_pct

Each tab calls the REAL pipeline (forecast.py, person3_commodity_forecast.py,
optimizer.py + sample_data.py, person5_risk_simulation.py) first. If a
teammate's piece isn't runnable in this environment yet (missing
statsmodels/xgboost/ortools, no database, missing forecast CSVs, etc.), the
tab falls back to the mock_* data below and says so visibly, instead of
silently showing fake numbers as if they were real.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import SUPPORTED_ROUTES
from pathlib import Path

import weather

ROOT = Path(__file__).resolve().parent


def _csv_row_count(filename: str):
    """Row count of one of Person 1's seed CSVs, or '—' if it's missing."""
    try:
        return len(pd.read_csv(ROOT / filename))
    except Exception:
        return "—"


def _default_budget() -> int:
    """Budget from sample_data.py so the optimizer tab starts feasible
    (the 3 sample cargoes cost ~$19M landed; a $5M default was always
    infeasible, so the tab silently fell back to demo data)."""
    try:
        from sample_data import get_sample_data
        return int(get_sample_data()[2]["budget"])
    except Exception:
        return 30_000_000


st.set_page_config(
    page_title="Maritime Freight & Chartering Optimizer",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 Freight Rate & Chartering Optimization Dashboard")
st.caption("SIH Prototype — Overview | Forecast | Optimizer | Weather & Cyclone Risk | Risk | ROI Summary")



# ===========================================================================
# CONTRACT-COMPATIBLE MOCK DATA
# ===========================================================================

def mock_forecast_freight_rate(route: str, horizon_weeks: int) -> pd.DataFrame:
    dates = pd.date_range(datetime.today(), periods=horizon_weeks, freq="W-MON")
    rng = np.random.default_rng(42)
    rate = 15000 + np.cumsum(rng.normal(50, 300, horizon_weeks))
    return pd.DataFrame(
        {
            "date": dates,
            "rate": rate,
            "ci_lower": rate - 800,
            "ci_upper": rate + 800,
        }
    )


def mock_forecast_commodity_price(commodity: str, origin: str, horizon_weeks: int) -> pd.DataFrame:
    dates = pd.date_range(datetime.today(), periods=horizon_weeks, freq="W-MON")
    base = {
        "coal_newcastle": 90,
        "wheat_gulf": 250,
        "corn_gulf": 190,
    }.get(commodity, 100)
    rng = np.random.default_rng(42)
    price = base + np.cumsum(rng.normal(0.3, 1.5, horizon_weeks))
    return pd.DataFrame(
        {
            "date": dates,
            "price": price,
            "ci_lower": price - 4,
            "ci_upper": price + 4,
        }
    )


def _cost_breakdown(total: float) -> str:
    return json.dumps(
        {
            "procurement_cost": round(total * 0.55, 2),
            "freight_cost": round(total * 0.25, 2),
            "vessel_fixed_cost": round(total * 0.10, 2),
            "port_cost": round(total * 0.10, 2),
            "total_cost": round(total, 2),
        }
    )


def mock_optimize(cargo_list, vessel_list, constraints) -> pd.DataFrame:
    """Mock matches Person 4's exact output contract."""
    totals = [1_200_000, 1_050_000, 890_000]
    return pd.DataFrame(
        {
            "cargo": ["C1", "C2", "C3"],
            "vessel": ["V1", "V2", "V3"],
            "port": ["Paradip", "Visakhapatnam", "Krishnapatnam"],
            "qty": [50_000, 45_000, 38_000],
            "cost_breakdown": [_cost_breakdown(total) for total in totals],
        }
    )


def mock_simulate_scenarios(n_scenarios: int) -> pd.DataFrame:
    """Mock matches Person 5's exact output contract and percentage units."""
    rng = np.random.default_rng(42)
    optimized_cost = 2_200_000 + rng.normal(0, 60_000, n_scenarios)
    baseline_cost = 2_650_000 + rng.normal(0, 60_000, n_scenarios)
    savings_pct = (baseline_cost - optimized_cost) / baseline_cost * 100
    return pd.DataFrame(
        {
            "scenario_id": range(n_scenarios),
            "optimized_cost": optimized_cost,
            "baseline_cost": baseline_cost,
            "savings_pct": savings_pct,
        }
    )


def optimizer_costs(plan_df: pd.DataFrame) -> pd.DataFrame:
    """Parse Person 4's JSON cost_breakdown for charts/metrics."""
    if "cost_breakdown" not in plan_df.columns:
        raise ValueError("Optimizer result is missing 'cost_breakdown'")

    out = plan_df.copy()
    parsed = out["cost_breakdown"].apply(json.loads)
    out["total_cost"] = parsed.apply(lambda item: float(item["total_cost"]))
    # Expected weather/cyclone-delay cost the optimizer priced into the port
    # choice (see weather.py). Older plans / the demo data don't have it.
    if parsed.apply(lambda item: "weather_risk_cost" in item).all():
        out["weather_risk_cost"] = parsed.apply(lambda item: float(item["weather_risk_cost"]))
    return out


# ===========================================================================
# API CLIENT
# ===========================================================================
# The dashboard prefers the FastAPI service (api.py). If it isn't running it
# falls back to importing the teammate modules in-process, so the demo works
# either way. Point it elsewhere with the API_URL environment variable.

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def _api(path: str, params: dict | None = None, payload: dict | None = None, timeout: int = 900):
    """Call the API. Returns parsed JSON, or None if the API isn't reachable
    (caller then falls back to in-process imports). Raises RuntimeError if the
    API *was* reached but rejected the request — that's a real pipeline error."""
    url = API_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", str(e))
        except Exception:
            detail = str(e)
        raise RuntimeError(detail)
    except Exception:
        return None  # service not running / not reachable


@st.cache_data(show_spinner=False, ttl=30)
def api_available() -> bool:
    return _api("/health") is not None


def _rows_to_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df
with st.sidebar:
    st.subheader("Backend")
    if api_available():
        st.success(f"API connected\n\n{API_URL}")
        st.caption(f"Docs: {API_URL}/docs")
    else:
        st.info("API not running — using in-process modules.\n\nStart it with:\n`python -m uvicorn api:app --port 8000`")

    try:
        from db import status as db_status
        st.caption(f"Data: {db_status()}")
    except Exception:
        pass


# ===========================================================================
# LIVE PIPELINE WIRING
# ===========================================================================
# Each live_* function tries the real teammate module and returns
# (result_df, error_message, source). error_message is None on success and
# source says whether it came from the API or from an in-process import.
# Results are cached so re-rendering the dashboard (Streamlit reruns the whole
# script on every widget interaction) doesn't repeat slow model fits/solver runs.

@st.cache_data(show_spinner=False)
def live_forecast_freight_rate(route: str, horizon_weeks: int):
    try:
        rows = _api("/forecast/freight", {"route": route, "weeks": horizon_weeks})
        if rows is not None:
            return _rows_to_df(rows), None, "API"
        from forecast import forecast_freight_rate
        return forecast_freight_rate(route, horizon_weeks), None, "in-process"
    except Exception as e:
        return None, str(e), "API" if api_available() else "in-process"


@st.cache_data(show_spinner=False)
def live_forecast_commodity_price(commodity: str, horizon_weeks: int):
    try:
        rows = _api("/forecast/commodity", {"commodity": commodity, "weeks": horizon_weeks})
        if rows is not None:
            return _rows_to_df(rows), None, "API"
        from person3_commodity_forecast import forecast_commodity_price
        return forecast_commodity_price(commodity, horizon_weeks), None, "in-process"
    except Exception as e:
        return None, str(e), "API" if api_available() else "in-process"


@st.cache_data(show_spinner=False)
def live_optimize(budget: float, use_real_fleet: bool = False):
    try:
        result = _api("/optimize", payload={"budget": budget, "use_real_fleet": use_real_fleet})
        if result is not None:
            plan = pd.DataFrame(result["plan"])
            # the API returns cost_breakdown as an object; the team contract
            # (and the charts below) use it as a JSON string
            plan["cost_breakdown"] = plan["cost_breakdown"].apply(json.dumps)
            return plan, None, "API"

        from optimizer import optimize
        if use_real_fleet:
            from fleet_data import get_fleet_inputs
            cargo_list, vessel_list, constraints = get_fleet_inputs(budget=budget)
        else:
            from sample_data import get_sample_data
            cargo_list, vessel_list, constraints = get_sample_data()
            constraints = {**constraints, "budget": budget}
        return optimize(cargo_list, vessel_list, constraints), None, "in-process"
    except Exception as e:
        return None, str(e), "API" if api_available() else "in-process"


@st.cache_data(show_spinner=False)
def live_simulate_scenarios(n_scenarios: int):
    try:
        result = _api("/simulate", payload={"n_scenarios": n_scenarios})
        if result is not None:
            return pd.DataFrame(result["scenarios"]), None, "API"
        from person5_risk_simulation import simulate_scenarios
        return simulate_scenarios(n_scenarios=n_scenarios), None, "in-process"
    except Exception as e:
        return None, str(e), "API" if api_available() else "in-process"


def use_live_or_mock(live_result, error, mock_df, label: str, source: str = "") -> pd.DataFrame:
    """Shows a one-line status caption and returns whichever data is usable."""
    if live_result is not None and not live_result.empty:
        st.caption(f"✅ Live: {label}" + (f" (via {source})" if source else ""))
        return live_result
    st.caption(f"⚠️ Demo data for {label} — live pipeline unavailable ({error}).")
    return mock_df


# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3, tab_weather, tab4, tab5 = st.tabs(
    ["📊 Overview", "📈 Forecast", "⚙️ Optimizer", "🌀 Weather & Cyclone Risk", "🎲 Risk", "💰 ROI Summary"]
)

with tab1:
    st.subheader("Project Overview")
    st.write(
        "This dashboard supports data-driven chartering and procurement "
        "decisions by combining freight rate forecasting, commodity price "
        "forecasting, an optimization engine, and risk simulation."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Routes", len(SUPPORTED_ROUTES))
    col2.metric("Vessels Tracked", _csv_row_count("vessels.csv"))
    col3.metric("Avg Forecasted Savings", "—")
    col4.metric("Ports Covered", _csv_row_count("ports.csv"))

    st.markdown("---")
    st.subheader("How it works")
    st.markdown(
        """
        1. **Forecast** tab — projected freight rates and commodity prices
        2. **Optimizer** tab — best cargo-to-vessel-to-port assignment plan
        3. **Risk** tab — Monte Carlo cost distribution vs a naive baseline
        4. **ROI Summary** tab — the headline savings number
        """
    )

with tab2:
    st.subheader("Freight Rate & Commodity Price Forecast")

    col_a, col_b = st.columns(2)
    with col_a:
        route = st.selectbox("Route", SUPPORTED_ROUTES)
        horizon = st.slider("Forecast horizon (weeks)", 4, 26, 12)
    with col_b:
        commodity = st.selectbox(
            "Commodity",
            ["coal_newcastle", "wheat_gulf", "corn_gulf"],
        )
        origin = st.selectbox(
            "Origin",
            {
                "coal_newcastle": ["Newcastle"],
                "wheat_gulf": ["US Gulf (New Orleans)"],
                "corn_gulf": ["US Gulf (New Orleans)"],
            }[commodity],
        )

    with st.spinner("Fetching freight & commodity forecasts..."):
        live_freight, freight_err, freight_src = live_forecast_freight_rate(route, horizon)
        live_price, price_err, price_src = live_forecast_commodity_price(commodity, horizon)

    freight_df = use_live_or_mock(
        live_freight, freight_err, mock_forecast_freight_rate(route, horizon),
        f"{route} freight forecast", freight_src,
    )
    price_df = use_live_or_mock(
        live_price, price_err, mock_forecast_commodity_price(commodity, origin, horizon),
        f"{commodity} price forecast", price_src,
    )

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(x=freight_df["date"], y=freight_df["rate"], name="Forecast")
    )
    fig1.add_trace(
        go.Scatter(
            x=freight_df["date"],
            y=freight_df["ci_upper"],
            name="Upper CI",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=freight_df["date"],
            y=freight_df["ci_lower"],
            name="Lower CI",
            fill="tonexty",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig1.update_layout(
        title=f"Freight Rate Forecast — {route}",
        xaxis_title="Date",
        yaxis_title="Freight Index / Rate",
    )
    st.plotly_chart(fig1, width="stretch")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=price_df["date"], y=price_df["price"], name="Forecast"))
    fig2.add_trace(
        go.Scatter(
            x=price_df["date"],
            y=price_df["ci_upper"],
            name="Upper CI",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=price_df["date"],
            y=price_df["ci_lower"],
            name="Lower CI",
            fill="tonexty",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig2.update_layout(
        title=f"{commodity} FOB Price Forecast — {origin}",
        xaxis_title="Date",
        yaxis_title="Price (USD/mt)",
    )
    st.plotly_chart(fig2, width="stretch")

with tab3:
    st.subheader("Cargo → Vessel → Port Optimization Plan")

    col_l, col_r = st.columns(2)
    with col_l:
        budget = st.number_input("Budget (USD)", value=_default_budget(), step=500_000)
    with col_r:
        fleet_choice = st.radio(
            "Fleet",
            ["Real fleet (from database)", "Sample (3 vessels)"],
            help="Real fleet: every vessel in Person 1's `vessels` table that is open "
                 "during the cargo laycan window, with freight priced from the forecast "
                 "index and the fixtures calibration (fleet_data.py).",
        )
    use_real_fleet = fleet_choice.startswith("Real")

    with st.spinner("Solving cargo → vessel → port assignment..."):
        live_plan, plan_err, plan_src = live_optimize(budget, use_real_fleet)

    plan_df = use_live_or_mock(
        live_plan, plan_err,
        mock_optimize(cargo_list=[], vessel_list=[], constraints={"budget": budget}),
        "optimizer plan", plan_src,
    )
    plan_df = optimizer_costs(plan_df)

    st.dataframe(plan_df, width="stretch")

    fig3 = px.bar(
        plan_df,
        x="cargo",
        y="total_cost",
        color="port",
        title="Cost by Cargo Assignment",
    )
    st.plotly_chart(fig3, width="stretch")

    st.metric("Total Optimized Cost", f"${plan_df['total_cost'].sum():,.0f}")
    if "weather_risk_cost" in plan_df.columns:
        st.caption(
            f"Of which ${plan_df['weather_risk_cost'].sum():,.0f} is expected "
            "weather/cyclone-delay risk cost baked into the port choice."
        )

with tab_weather:
    st.subheader("🌀 Weather & Cyclone Risk")
    st.write(
        "Bay of Bengal cyclone-season climatology, folded into the optimizer "
        "as an expected-cost adder per (cargo, port) laycan window, a hard "
        "block above a severe-risk threshold, and a seasonal regressor for "
        "Person 2/3's forecasting models. See `weather.py` for the full "
        "real-vs-synthetic breakdown of what's calibrated to actual IMD "
        "climatology vs a planning-stage assumption."
    )

    climatology = weather.monthly_climatology_table()
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig_weather = px.line(
        climatology,
        x="month_name",
        y="risk_score",
        color="port",
        category_orders={"month_name": month_order},
        markers=True,
        title="Cyclone Risk Score by Port and Month (climatology)",
        labels={"month_name": "Month", "risk_score": "Risk score (0-1)"},
    )
    fig_weather.add_hline(
        y=1.0, line_dash="dot", line_color="red",
        annotation_text="cyclone hard-block ceiling", annotation_position="top left",
    )
    st.plotly_chart(fig_weather, width="stretch")

    st.markdown("---")
    st.subheader("Risk for a specific cargo laycan window")

    col1, col2, col3 = st.columns(3)
    with col1:
        wx_port = st.selectbox("Port", list(weather.PORT_CYCLONE_EXPOSURE.keys()))
    with col2:
        wx_start = st.date_input("Laycan start", value=datetime(2026, 10, 20))
    with col3:
        wx_end = st.date_input("Laycan end", value=datetime(2026, 11, 5))

    wx_risk = weather.window_risk(wx_port, wx_start, wx_end)
    wx_delay = weather.expected_delay_days(wx_port, wx_start, wx_end)
    wx_cost = weather.weather_cost_adder(wx_port, wx_start, wx_end)
    wx_blocked = weather.is_extreme_risk(wx_port, wx_start, wx_end)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Worst-day risk score", f"{wx_risk:.2f}")
    c2.metric("Expected delay", f"{wx_delay:.1f} days")
    c3.metric("Weather-risk cost", f"${wx_cost:,.0f}")
    c4.metric("Hard-blocked?", "🚫 Yes" if wx_blocked else "✅ No")

    if wx_blocked:
        st.error(
            f"{wx_port} in this laycan window is above the cyclone hard-block "
            "threshold — the optimizer will not route cargo there for these dates."
        )

with tab4:
    st.subheader("Monte Carlo Risk Simulation")

    n_scenarios = st.slider("Number of scenarios", 100, 5000, 1000, step=100)

    with st.spinner(f"Running {n_scenarios:,} Monte Carlo scenarios..."):
        live_sim, sim_err, sim_src = live_simulate_scenarios(n_scenarios)

    sim_df = use_live_or_mock(
        live_sim, sim_err, mock_simulate_scenarios(n_scenarios), "Monte Carlo risk simulation", sim_src,
    ).dropna()

    fig4 = go.Figure()
    fig4.add_trace(
        go.Histogram(
            x=sim_df["baseline_cost"],
            name="Baseline (no optimization)",
            opacity=0.6,
        )
    )
    fig4.add_trace(
        go.Histogram(
            x=sim_df["optimized_cost"],
            name="Optimized",
            opacity=0.6,
        )
    )
    fig4.update_layout(
        barmode="overlay",
        title="Cost Distribution: Optimized vs Baseline",
        xaxis_title="Total Cost (USD)",
        yaxis_title="Scenario Count",
    )
    st.plotly_chart(fig4, width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Optimized Cost", f"${sim_df['optimized_cost'].mean():,.0f}")
    col2.metric("Avg Baseline Cost", f"${sim_df['baseline_cost'].mean():,.0f}")
    col3.metric("Avg Savings", f"{sim_df['savings_pct'].mean():.1f}%")

with tab5:
    st.subheader("💰 ROI Summary — The Headline Pitch")

    with st.spinner("Running Monte Carlo scenarios for the ROI summary..."):
        live_sim, sim_err, sim_src = live_simulate_scenarios(1000)

    sim_df = use_live_or_mock(
        live_sim, sim_err, mock_simulate_scenarios(1000), "Monte Carlo risk simulation", sim_src,
    ).dropna()
    avg_savings_pct = sim_df["savings_pct"].mean()
    avg_savings_usd = (sim_df["baseline_cost"] - sim_df["optimized_cost"]).mean()

    st.markdown(
        f"""
        ### Our optimization model delivers an average
        # **{avg_savings_pct:.1f}% cost reduction**
        ### compared to a naive spot-chartering baseline
        """
    )

    col1, col2 = st.columns(2)
    col1.metric("Avg Savings per Voyage", f"${avg_savings_usd:,.0f}")
    col2.metric("Confidence (scenarios simulated)", f"{len(sim_df):,}")

    st.markdown("---")
    st.subheader("Why this matters")
    st.markdown(
        """
        - Data-driven chartering decisions instead of ad-hoc spot booking
        - Forecasting reduces exposure to freight rate volatility
        - Monte Carlo simulation quantifies risk, not just point estimates
        - Optimizer respects DWT, port-draft, laycan, and budget constraints
        """
    )

    fig5 = px.box(
        sim_df,
        y=["baseline_cost", "optimized_cost"],
        title="Cost Spread: Baseline vs Optimized",
    )
    st.plotly_chart(fig5, width="stretch")
