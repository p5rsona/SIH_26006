"""
dashboard.py
============
Person 6 — Streamlit Dashboard & Integration Lead

Run with:  streamlit run dashboard.py

All four teammate modules are now wired in for real:
  - Person 2: forecast.py            -> forecast_freight_rate()
  - Person 3: person3_commodity_forecast.py -> forecast_commodity_price()
  - Person 4: optimizer.py           -> optimize()
  - Person 5: person5_risk_simulation.py    -> simulate_scenarios()

Heads up on startup time: importing person3_commodity_forecast.py trains
a SARIMAX + XGBoost model for every commodity as soon as it's imported
(it's not wrapped in a function at the top level), so the app may take
10-30 seconds to start the first time. That's expected, not a bug on
your end.
"""

import json

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import SUPPORTED_ROUTES
from forecast import forecast_freight_rate
from person3_commodity_forecast import forecast_commodity_price
from optimizer import optimize
from sample_data import get_sample_data
from person5_risk_simulation import simulate_scenarios

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

# Person 3's commodity codes, as used inside her forecast_*.csv files / df["commodity"]
COMMODITY_OPTIONS = ["coal_newcastle", "wheat_gulf", "corn_gulf"]

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

    cargo_list, vessel_list, constraints = get_sample_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Routes Supported", len(SUPPORTED_ROUTES))
    col2.metric("Vessels (sample data)", len(vessel_list))
    col3.metric("Cargo Lots (sample data)", len(cargo_list))
    col4.metric("Ports (sample data)", len(constraints["ports"]))

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
        route = st.selectbox("Route", SUPPORTED_ROUTES)
        horizon = st.slider("Forecast horizon (weeks)", 4, 26, 12)
    with colB:
        commodity = st.selectbox("Commodity", COMMODITY_OPTIONS)
        model_choice = st.radio("Freight model", ["sarimax", "xgboost"], horizontal=True)

    with st.spinner("Forecasting freight rate..."):
        freight_df = forecast_freight_rate(route, horizon, model=model_choice)

    with st.spinner("Forecasting commodity price..."):
        price_df = forecast_commodity_price(commodity, horizon)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=freight_df["date"], y=freight_df["rate"], name="Forecast", line=dict(color="royalblue")))
    fig1.add_trace(go.Scatter(x=freight_df["date"], y=freight_df["ci_upper"], name="Upper CI", line=dict(width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=freight_df["date"], y=freight_df["ci_lower"], name="Lower CI", fill="tonexty", line=dict(width=0), fillcolor="rgba(65,105,225,0.2)", showlegend=False))
    fig1.update_layout(title=f"Freight Rate Forecast — {route} ({model_choice})", xaxis_title="Date", yaxis_title="Rate (USD/day)")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=price_df["date"], y=price_df["price"], name="Forecast", line=dict(color="seagreen")))
    fig2.add_trace(go.Scatter(x=price_df["date"], y=price_df["ci_upper"], line=dict(width=0), showlegend=False))
    fig2.add_trace(go.Scatter(x=price_df["date"], y=price_df["ci_lower"], fill="tonexty", line=dict(width=0), fillcolor="rgba(46,139,87,0.2)", showlegend=False))
    fig2.update_layout(title=f"{commodity} FOB Price Forecast", xaxis_title="Date", yaxis_title="Price (USD/tonne)")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 — Optimizer
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Cargo → Vessel → Port Optimization Plan")

    cargo_list, vessel_list, constraints = get_sample_data()

    budget = st.number_input(
        "Budget (USD)",
        value=int(constraints["budget"]),
        step=100000,
    )
    constraints = {**constraints, "budget": budget}

    with st.spinner("Solving optimization model..."):
        try:
            plan_df = optimize(cargo_list, vessel_list, constraints)
            plan_df["total_cost"] = plan_df["cost_breakdown"].apply(lambda s: json.loads(s)["total_cost"])

            st.dataframe(plan_df, use_container_width=True)

            fig3 = px.bar(plan_df, x="cargo", y="total_cost", color="port", title="Cost by Cargo Assignment")
            st.plotly_chart(fig3, use_container_width=True)

            st.metric("Total Optimized Cost", f"${plan_df['total_cost'].sum():,.0f}")
        except RuntimeError as e:
            st.error(f"No feasible plan found with this budget: {e}")

# ---------------------------------------------------------------------------
# TAB 4 — Risk
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Monte Carlo Risk Simulation")
    st.caption(
        "Each scenario re-solves the full optimizer, so this can take a "
        "little while for larger scenario counts."
    )

    n_scenarios = st.slider("Number of scenarios", 10, 300, 50, step=10)
    run_sim = st.button("Run simulation")

    if run_sim or "sim_df" not in st.session_state:
        with st.spinner(f"Running {n_scenarios} Monte Carlo scenarios..."):
            st.session_state["sim_df"] = simulate_scenarios(n_scenarios=n_scenarios)

    sim_df = st.session_state["sim_df"].dropna()

    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=sim_df["baseline_cost"], name="Baseline (no optimization)", opacity=0.6, marker_color="indianred"))
    fig4.add_trace(go.Histogram(x=sim_df["optimized_cost"], name="Optimized", opacity=0.6, marker_color="royalblue"))
    fig4.update_layout(barmode="overlay", title="Cost Distribution: Optimized vs Baseline", xaxis_title="Total Cost (USD)", yaxis_title="Scenario Count")
    st.plotly_chart(fig4, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Optimized Cost", f"${sim_df['optimized_cost'].mean():,.0f}")
    col2.metric("Avg Baseline Cost", f"${sim_df['baseline_cost'].mean():,.0f}")
    col3.metric("Avg Savings", f"{sim_df['savings_pct'].mean():.1f}%")

# ---------------------------------------------------------------------------
# TAB 5 — ROI Summary
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("💰 ROI Summary — The Headline Pitch")

    if "sim_df" not in st.session_state:
        st.info("Run a simulation in the Risk tab first to see the ROI summary here.")
    else:
        sim_df = st.session_state["sim_df"].dropna()
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
        col2.metric("Scenarios Simulated", f"{len(sim_df):,}")

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

        fig5 = px.box(sim_df, y=["baseline_cost", "optimized_cost"], title="Cost Spread: Baseline vs Optimized")
        st.plotly_chart(fig5, use_container_width=True)
