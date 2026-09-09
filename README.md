# Person 4 – Optimization Engine

## Objective

The Optimization Engine determines a feasible and cost-efficient
chartering plan by assigning cargo lots to vessels and ports while
respecting operational constraints.

## Technology

- Python
- Google OR-Tools CP-SAT
- Pandas

## Inputs

The optimizer accepts:

- Cargo information
- Vessel information
- Port information
- Procurement prices
- Freight costs
- Vessel costs
- Port costs
- Laycan windows
- Vessel availability
- Vessel DWT
- Vessel draft
- Overall budget

## Optimization Constraints

The model considers:

1. Cargo must be assigned to a feasible vessel and port.
2. Vessel DWT cannot be exceeded.
3. Vessel draft cannot exceed the port's maximum draft.
4. Cargo laycan must overlap with vessel availability.
5. Optional allowed-port restrictions are supported.
6. Total cost must remain within the specified budget.

## Objective Function

The optimizer minimizes total landed cost:

Procurement Cost
+ Freight Cost
+ Vessel Fixed Cost
+ Port Cost

## Output

The optimizer returns a Pandas DataFrame containing:

- cargo
- vessel
- port
- qty
- cost_breakdown

## Test Data

The initial implementation uses static vessel, cargo and port data so
that the optimization engine can be developed independently of the
database and forecasting modules.

Forecast commodity prices are read from the supplied forecast CSV files.

## Running the Program

Install dependencies:

    python -m pip install -r requirements.txt

Run:

    python optimizer.py

The program prints the optimal chartering plan.

## Integration

The optimizer is designed so that the static test data can later be
replaced with data supplied by the team's database/data-engineering
module and forecast outputs from the forecasting modules.