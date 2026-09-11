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
statsmodels/xgboost/ortools, no MySQL, missing forecast CSVs, etc.), the
tab falls back to the mock_* data below and says so visibly, instead of
silently showing fake numbers as if they were real.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import SUPPORTED_ROUTES
from pathlib import Path

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
st.caption("SIH Prototype — Overview | Forecast | Optimizer | Risk | ROI Summary")


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
    return out


# ===========================================================================
# LIVE PIPELINE WIRING
# ===========================================================================
# Each live_* function tries the real teammate module and returns
# (result_df, error_message). error_message is None on success. Results are
# cached so re-rendering the dashboard (Streamlit reruns the whole script on
# every widget interaction) doesn't repeat slow model fits / solver runs.

@st.cache_data(show_spinner=False)
def live_forecast_freight_rate(route: str, horizon_weeks: int):
    try:
        from forecast import forecast_freight_rate
        return forecast_freight_rate(route, horizon_weeks), None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def live_forecast_commodity_price(commodity: str, horizon_weeks: int):
    try:
        from person3_commodity_forecast import forecast_commodity_price
        return forecast_commodity_price(commodity, horizon_weeks), None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def live_optimize(budget: float):
    try:
        from optimizer import optimize
        from sample_data import get_sample_data
        cargo_list, vessel_list, constraints = get_sample_data()
        constraints = {**constraints, "budget": budget}
        return optimize(cargo_list, vessel_list, constraints), None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def live_simulate_scenarios(n_scenarios: int):
    try:
        from person5_risk_simulation import simulate_scenarios
        return simulate_scenarios(n_scenarios=n_scenarios), None
    except Exception as e:
        return None, str(e)


def use_live_or_mock(live_result, error, mock_df, label: str) -> pd.DataFrame:
    """Shows a one-line status caption and returns whichever data is usable."""
    if live_result is not None and not live_result.empty:
        st.caption(f"✅ Live: {label}")
        return live_result
    st.caption(f"⚠️ Demo data for {label} — live pipeline unavailable ({error}).")
    return mock_df


# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📈 Forecast", "⚙️ Optimizer", "🎲 Risk", "💰 ROI Summary"]
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
        live_freight, freight_err = live_forecast_freight_rate(route, horizon)
        live_price, price_err = live_forecast_commodity_price(commodity, horizon)

    freight_df = use_live_or_mock(
        live_freight, freight_err, mock_forecast_freight_rate(route, horizon),
        f"{route} freight forecast",
    )
    price_df = use_live_or_mock(
        live_price, price_err, mock_forecast_commodity_price(commodity, origin, horizon),
        f"{commodity} price forecast",
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

    budget = st.number_input("Budget (USD)", value=_default_budget(), step=500_000)

    with st.spinner("Solving cargo → vessel → port assignment..."):
        live_plan, plan_err = live_optimize(budget)

    plan_df = use_live_or_mock(
        live_plan, plan_err,
        mock_optimize(cargo_list=[], vessel_list=[], constraints={"budget": budget}),
        "optimizer plan",
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

with tab4:
    st.subheader("Monte Carlo Risk Simulation")

    n_scenarios = st.slider("Number of scenarios", 100, 5000, 1000, step=100)

    with st.spinner(f"Running {n_scenarios:,} Monte Carlo scenarios..."):
        live_sim, sim_err = live_simulate_scenarios(n_scenarios)

    sim_df = use_live_or_mock(
        live_sim, sim_err, mock_simulate_scenarios(n_scenarios), "Monte Carlo risk simulation",
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
        live_sim, sim_err = live_simulate_scenarios(1000)

    sim_df = use_live_or_mock(
        live_sim, sim_err, mock_simulate_scenarios(1000), "Monte Carlo risk simulation",
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
