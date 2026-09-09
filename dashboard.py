"""
dashboard.py
============
Person 6 — Streamlit Dashboard & Integration Lead

Run with:  streamlit run dashboard.py

This app works standalone right now using mock data generators at the
bottom of the file. As teammates finish their modules, replace the
mock_* calls in each tab with the real functions (see the "SWAP HERE"
comments) — e.g. once Person 2 finishes, swap mock_forecast_freight_rate
for their real forecast_freight_rate function.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Maritime Freight & Chartering Optimizer",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 Freight Rate & Chartering Optimization Dashboard")
st.caption("SIH Prototype — Overview | Forecast | Optimizer | Risk | ROI Summary")


# ===========================================================================
# MOCK DATA GENERATORS
# Replace these with real function calls once teammates' modules are ready.
# ===========================================================================

def mock_forecast_freight_rate(route: str, horizon_weeks: int) -> pd.DataFrame:
    dates = pd.date_range(datetime.today(), periods=horizon_weeks, freq="W")
    base = 15000
    rate = base + np.cumsum(np.random.normal(50, 300, horizon_weeks))
    return pd.DataFrame({
        "date": dates,
        "rate": rate,
        "ci_lower": rate - 800,
        "ci_upper": rate + 800,
    })


def mock_forecast_commodity_price(commodity: str, origin: str, horizon_weeks: int) -> pd.DataFrame:
    dates = pd.date_range(datetime.today(), periods=horizon_weeks, freq="W")
    base = 90
    price = base + np.cumsum(np.random.normal(0.3, 1.5, horizon_weeks))
    return pd.DataFrame({
        "date": dates,
        "price": price,
        "ci_lower": price - 4,
        "ci_upper": price + 4,
    })


def mock_optimize(cargo_list, vessel_list, constraints) -> pd.DataFrame:
    return pd.DataFrame({
        "cargo_id": ["C1", "C2", "C3"],
        "vessel_id": ["V1", "V2", "V3"],
        "port": ["Paradip", "Vizag", "Krishnapatnam"],
        "qty_mt": [50000, 45000, 38000],
        "cost_usd": [1_200_000, 1_050_000, 890_000],
    })


def mock_simulate_scenarios(n_scenarios: int) -> pd.DataFrame:
    total_cost = 2_200_000 + np.random.normal(0, 60000, n_scenarios)
    baseline_cost = 2_650_000 + np.random.normal(0, 60000, n_scenarios)
    return pd.DataFrame({
        "scenario_id": range(n_scenarios),
        "total_cost": total_cost,
        "baseline_cost": baseline_cost,
        "savings_pct": (baseline_cost - total_cost) / baseline_cost,
    })


# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📈 Forecast", "⚙️ Optimizer", "🎲 Risk", "💰 ROI Summary"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Overview
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Project Overview")
    st.write(
        "This dashboard supports data-driven chartering and procurement "
        "decisions by combining freight rate forecasting, commodity price "
        "forecasting, an optimization engine, and risk simulation."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Routes", "12")
    col2.metric("Vessels Tracked", "34")
    col3.metric("Avg Forecasted Savings", "15.2%")
    col4.metric("Ports Covered", "8")

    st.markdown("---")
    st.subheader("How it works")
    st.markdown(
        """
        1. **Forecast** tab — projected freight rates and commodity prices
        2. **Optimizer** tab — best cargo-to-vessel-to-port assignment plan
        3. **Risk** tab — Monte Carlo cost distribution vs a naive baseline
        4. **ROI Summary** tab — the headline savings number for judges
        """
    )

# ---------------------------------------------------------------------------
# TAB 2 — Forecast
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Freight Rate & Commodity Price Forecast")

    colA, colB = st.columns(2)
    with colA:
        route = st.selectbox("Route", ["Australia-EastCoastIndia", "Indonesia-EastCoastIndia", "SouthAfrica-EastCoastIndia"])
        horizon = st.slider("Forecast horizon (weeks)", 4, 26, 12)
    with colB:
        commodity = st.selectbox("Commodity", ["coal", "grain", "iron_ore"])
        origin = st.selectbox("Origin", ["Newcastle", "Kalimantan", "Richards Bay"])

    # SWAP HERE: replace mock_forecast_freight_rate with Person 2's real function
    freight_df = mock_forecast_freight_rate(route, horizon)
    # SWAP HERE: replace mock_forecast_commodity_price with Person 3's real function
    price_df = mock_forecast_commodity_price(commodity, origin, horizon)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=freight_df["date"], y=freight_df["rate"], name="Forecast", line=dict(color="royalblue")))
    fig1.add_trace(go.Scatter(x=freight_df["date"], y=freight_df["ci_upper"], name="Upper CI", line=dict(width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=freight_df["date"], y=freight_df["ci_lower"], name="Lower CI", fill="tonexty", line=dict(width=0), fillcolor="rgba(65,105,225,0.2)", showlegend=False))
    fig1.update_layout(title=f"Freight Rate Forecast — {route}", xaxis_title="Date", yaxis_title="Rate (USD/day)")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=price_df["date"], y=price_df["price"], name="Forecast", line=dict(color="seagreen")))
    fig2.add_trace(go.Scatter(x=price_df["date"], y=price_df["ci_upper"], line=dict(width=0), showlegend=False))
    fig2.add_trace(go.Scatter(x=price_df["date"], y=price_df["ci_lower"], fill="tonexty", line=dict(width=0), fillcolor="rgba(46,139,87,0.2)", showlegend=False))
    fig2.update_layout(title=f"{commodity.title()} FOB Price Forecast — {origin}", xaxis_title="Date", yaxis_title="Price (USD/mt)")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — Optimizer
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Cargo → Vessel → Port Optimization Plan")

    budget = st.number_input("Budget (USD)", value=5_000_000, step=100000)

    # SWAP HERE: replace mock_optimize with Person 4's real optimize() function
    plan_df = mock_optimize(cargo_list=[], vessel_list=[], constraints={"budget_usd": budget})

    st.dataframe(plan_df, use_container_width=True)

    fig3 = px.bar(plan_df, x="cargo_id", y="cost_usd", color="port", title="Cost by Cargo Assignment")
    st.plotly_chart(fig3, use_container_width=True)

    st.metric("Total Optimized Cost", f"${plan_df['cost_usd'].sum():,.0f}")

# ---------------------------------------------------------------------------
# TAB 4 — Risk
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Monte Carlo Risk Simulation")

    n_scenarios = st.slider("Number of scenarios", 100, 5000, 1000, step=100)

    # SWAP HERE: replace mock_simulate_scenarios with Person 5's real function
    sim_df = mock_simulate_scenarios(n_scenarios)

    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=sim_df["baseline_cost"], name="Baseline (no optimization)", opacity=0.6, marker_color="indianred"))
    fig4.add_trace(go.Histogram(x=sim_df["total_cost"], name="Optimized", opacity=0.6, marker_color="royalblue"))
    fig4.update_layout(barmode="overlay", title="Cost Distribution: Optimized vs Baseline", xaxis_title="Total Cost (USD)", yaxis_title="Scenario Count")
    st.plotly_chart(fig4, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Optimized Cost", f"${sim_df['total_cost'].mean():,.0f}")
    col2.metric("Avg Baseline Cost", f"${sim_df['baseline_cost'].mean():,.0f}")
    col3.metric("Avg Savings", f"{sim_df['savings_pct'].mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# TAB 5 — ROI Summary
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("💰 ROI Summary — The Headline Pitch")

    # Reuse the risk simulation for the headline numbers
    sim_df = mock_simulate_scenarios(1000)
    avg_savings_pct = sim_df["savings_pct"].mean() * 100
    avg_savings_usd = (sim_df["baseline_cost"] - sim_df["total_cost"]).mean()

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
        - Optimizer respects real-world constraints: DWT capacity, port draft, laycan windows
        """
    )

    fig5 = px.box(sim_df, y=["baseline_cost", "total_cost"], title="Cost Spread: Baseline vs Optimized")
    st.plotly_chart(fig5, use_container_width=True)
