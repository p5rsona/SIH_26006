"""
api.py — REST API in front of the whole pipeline (Person 6 / integration).

Run it:
    python -m uvicorn api:app --reload --port 8000

Then open http://localhost:8000/docs for interactive documentation where you
can try every endpoint from the browser.

This file is a thin wrapper: it does not reimplement anything. Each endpoint
calls the teammate module that already owns that logic:

    /forecast/freight    -> forecast.py                  (Person 2)
    /forecast/commodity  -> person3_commodity_forecast.py (Person 3)
    /optimize            -> optimizer.py                  (Person 4)
    /simulate            -> person5_risk_simulation.py    (Person 5)
    /weather/*           -> weather.py                     (weather & cyclone risk)

The Pydantic models below are the team's interface contracts written as code:
if a module ever returns a different shape, the API fails loudly here instead
of silently breaking the dashboard.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from config import SUPPORTED_ROUTES

COMMODITIES = ["coal_newcastle", "wheat_gulf", "corn_gulf"]

app = FastAPI(
    title="Maritime Procurement & Chartering Optimizer API",
    description=(
        "Freight rate forecasting, commodity price forecasting, cargo-to-vessel-to-port "
        "optimization and Monte Carlo risk simulation for dry bulk imports to east-coast India."
    ),
    version="1.0.0",
)


# =====================================================================
# Response models — the team's output contracts, enforced
# =====================================================================

class HealthResponse(BaseModel):
    status: str = "ok"
    database: str = Field(default="", description="Backend and whether it is reachable")
    modules: Dict[str, bool] = Field(
        description="Which teammate modules imported successfully in this process"
    )


class MetaResponse(BaseModel):
    routes: List[str]
    commodities: List[str]
    models: List[str]


class FreightPoint(BaseModel):
    date: str = Field(examples=["2026-08-31"])
    rate: float = Field(description="Forecast freight rate (Baltic index points)")
    ci_lower: float
    ci_upper: float


class CommodityPoint(BaseModel):
    date: str
    price: float = Field(description="Forecast FOB price, USD per tonne")
    ci_lower: float
    ci_upper: float


class PlanRow(BaseModel):
    cargo: str
    vessel: str
    port: str
    qty: int
    cost_breakdown: Dict[str, float] = Field(
        description="procurement_cost, freight_cost, vessel_fixed_cost, port_cost, "
                    "weather_risk_cost, total_cost"
    )


class OptimizeResponse(BaseModel):
    plan: List[PlanRow]
    total_cost: float


class OptimizeRequest(BaseModel):
    """Leave every field out to optimize the sample cargoes from sample_data.py."""
    cargo_list: Optional[List[Dict[str, Any]]] = None
    vessel_list: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[Dict[str, Any]] = None
    budget: Optional[float] = Field(
        default=None, description="Overrides constraints['budget'] when given"
    )
    use_real_fleet: bool = Field(
        default=False,
        description="Build the inputs from the real ports/vessels/routes tables "
                    "(fleet_data.py) instead of the 3 hardcoded sample vessels",
    )
    laycan_start: str = "2026-08-24"
    laycan_end: str = "2026-09-15"


class VesselOut(BaseModel):
    vessel: str
    vessel_class: str
    dwt: int
    draft: float
    available_start: str
    available_end: str


class SimulateRequest(BaseModel):
    n_scenarios: int = Field(default=200, ge=1, le=5000)
    route: str = "C5"
    horizon_weeks: int = Field(default=8, ge=1, le=52)
    seed: int = 42


class ScenarioRow(BaseModel):
    scenario_id: int
    optimized_cost: Optional[float]
    baseline_cost: Optional[float]
    savings_pct: Optional[float]
    weather_multiplier: Optional[float] = Field(
        default=None,
        description="How much worse/better than climatology weather turned out "
                    "in this scenario (1.0 = as expected)",
    )


class WeatherRiskResponse(BaseModel):
    port: str
    laycan_start: str
    laycan_end: str
    risk_score: float = Field(description="Worst-day cyclone risk over the window, 0-1")
    expected_delay_days: float
    weather_risk_cost: float = Field(description="Expected delay priced at the demurrage rate, USD")
    hard_blocked: bool = Field(
        description="True if the optimizer refuses to route cargo to this port for these dates"
    )


class ClimatologyRow(BaseModel):
    port: str
    month: int
    month_name: str
    risk_score: float


class SimulateResponse(BaseModel):
    summary: Dict[str, float]
    scenarios: List[ScenarioRow]


# =====================================================================
# Helpers
# =====================================================================

def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame -> JSON-safe records, with dates as YYYY-MM-DD strings."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")
    return json.loads(out.to_json(orient="records"))


def _fail(exc: Exception, what: str):
    """ValueError = the caller asked for something invalid (400);
    anything else = the pipeline broke (500)."""
    if isinstance(exc, (ValueError, KeyError)):
        raise HTTPException(status_code=400, detail=str(exc))
    raise HTTPException(status_code=500, detail=f"{what} failed: {exc}")


# =====================================================================
# Endpoints
# =====================================================================

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness check, plus which teammate modules can be imported here."""
    modules = {}
    for name, target in [
        ("forecast_freight", "forecast"),
        ("forecast_commodity", "person3_commodity_forecast"),
        ("optimizer", "optimizer"),
        ("risk_simulation", "person5_risk_simulation"),
        ("weather", "weather"),
    ]:
        try:
            __import__(target)
            modules[name] = True
        except Exception:
            modules[name] = False
    try:
        from db import status as db_status
        database = db_status()
    except Exception as e:
        database = f"unavailable ({e})"
    return HealthResponse(status="ok", database=database, modules=modules)


@app.get("/meta", response_model=MetaResponse, tags=["meta"])
def meta() -> MetaResponse:
    """Valid values for the route, commodity and model parameters."""
    return MetaResponse(
        routes=list(SUPPORTED_ROUTES), commodities=COMMODITIES, models=["sarimax", "xgboost"]
    )


@app.get("/forecast/freight", response_model=List[FreightPoint], tags=["forecast"])
def freight_forecast(
    route: str = Query("C5", description=f"One of {SUPPORTED_ROUTES}"),
    weeks: int = Query(8, ge=1, le=52, description="Forecast horizon in weeks"),
    model: str = Query("sarimax", pattern="^(sarimax|xgboost)$"),
):
    """Person 2's freight rate forecast: date, rate, ci_lower, ci_upper."""
    try:
        from forecast import forecast_freight_rate
        return _records(forecast_freight_rate(route, weeks, model=model))
    except Exception as e:
        _fail(e, "freight forecast")


@app.get("/forecast/commodity", response_model=List[CommodityPoint], tags=["forecast"])
def commodity_forecast(
    commodity: str = Query("coal_newcastle", description=f"One of {COMMODITIES}"),
    weeks: int = Query(4, ge=1, le=52),
):
    """Person 3's FOB commodity price forecast: date, price, ci_lower, ci_upper."""
    try:
        from person3_commodity_forecast import forecast_commodity_price
        return _records(forecast_commodity_price(commodity, weeks))
    except Exception as e:
        _fail(e, "commodity forecast")


@app.post("/optimize", response_model=OptimizeResponse, tags=["optimize"])
def optimize_plan(request: OptimizeRequest = OptimizeRequest()):
    """Person 4's optimizer: cheapest cargo -> vessel -> port assignment,
    respecting DWT, port draft, laycan windows and budget."""
    try:
        from optimizer import optimize
        from sample_data import get_sample_data

        if request.use_real_fleet:
            from fleet_data import get_fleet_inputs
            cargo_list, vessel_list, constraints = get_fleet_inputs(
                laycan_start=request.laycan_start, laycan_end=request.laycan_end,
                budget=request.budget,
            )
        else:
            cargo_list, vessel_list, constraints = get_sample_data()
        if request.cargo_list is not None:
            cargo_list = request.cargo_list
        if request.vessel_list is not None:
            vessel_list = request.vessel_list
        if request.constraints is not None:
            constraints = request.constraints
        if request.budget is not None:
            constraints = {**constraints, "budget": request.budget}

        plan = optimize(cargo_list, vessel_list, constraints)
        rows = [
            PlanRow(
                cargo=r["cargo"], vessel=r["vessel"], port=r["port"],
                qty=int(r["qty"]), cost_breakdown=json.loads(r["cost_breakdown"]),
            )
            for r in plan.to_dict(orient="records")
        ]
        return OptimizeResponse(
            plan=rows, total_cost=sum(r.cost_breakdown["total_cost"] for r in rows)
        )
    except RuntimeError as e:
        # optimizer raises this when no plan satisfies the constraints
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        _fail(e, "optimization")


@app.get("/fleet", response_model=List[VesselOut], tags=["optimize"])
def fleet(
    laycan_start: str = Query("2026-08-24"),
    laycan_end: str = Query("2026-09-15"),
):
    """Vessels from Person 1's fleet that are open during the laycan window."""
    try:
        from fleet_data import get_fleet_inputs
        _, vessel_list, _ = get_fleet_inputs(laycan_start=laycan_start, laycan_end=laycan_end)
        return [VesselOut(**{k: v[k] for k in VesselOut.__annotations__}) for v in vessel_list]
    except Exception as e:
        _fail(e, "fleet lookup")


@app.get("/weather/risk", response_model=WeatherRiskResponse, tags=["weather"])
def weather_risk(
    port: str = Query("Paradip", description="Destination port, e.g. Paradip / Visakhapatnam / Kakinada / Krishnapatnam"),
    laycan_start: str = Query("2026-10-20"),
    laycan_end: str = Query("2026-11-05"),
):
    """Cyclone/weather risk for one port and laycan window — the same numbers
    the optimizer prices into its port choice (weather.py)."""
    try:
        import weather

        return WeatherRiskResponse(
            port=port,
            laycan_start=laycan_start,
            laycan_end=laycan_end,
            risk_score=weather.window_risk(port, laycan_start, laycan_end),
            expected_delay_days=weather.expected_delay_days(port, laycan_start, laycan_end),
            weather_risk_cost=weather.weather_cost_adder(port, laycan_start, laycan_end),
            hard_blocked=weather.is_extreme_risk(port, laycan_start, laycan_end),
        )
    except Exception as e:
        _fail(e, "weather risk")


@app.get("/weather/climatology", response_model=List[ClimatologyRow], tags=["weather"])
def weather_climatology():
    """Monthly cyclone-risk climatology per port, for charting."""
    try:
        import weather

        return _records(weather.monthly_climatology_table())
    except Exception as e:
        _fail(e, "weather climatology")


@app.post("/simulate", response_model=SimulateResponse, tags=["risk"])
def simulate(request: SimulateRequest = SimulateRequest()):
    """Person 5's Monte Carlo: optimized vs naive baseline cost distribution.

    Note: a few thousand scenarios take a while — the request stays open until
    it finishes. Background jobs are the planned next step.
    """
    try:
        from person5_risk_simulation import simulate_scenarios, summarize

        df = simulate_scenarios(
            n_scenarios=request.n_scenarios,
            route=request.route,
            horizon_weeks=request.horizon_weeks,
            seed=request.seed,
        )
        summary = {k: float(v) for k, v in summarize(df).items()}
        return SimulateResponse(
            summary=summary,
            scenarios=[ScenarioRow(**row) for row in _records(df.replace({float("nan"): None}))],
        )
    except Exception as e:
        _fail(e, "risk simulation")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
