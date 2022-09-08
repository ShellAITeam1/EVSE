"""Main module."""
import logging
from pathlib import Path

import pandas as pd
from ortools.linear_solver import pywraplp

from evse.const import FAST_CHARGING_STATION_TAG, SLOW_CHARGING_STATION_TAG
from evse.optimisation import create_data_model
from evse.optimisation._modelisation import DataModel

MAX_INCREASING_CAPACITIES = 10
MAXIMUM_SUPPLY_PROPORTION_AVAILABLE = 100


def main(data_model: DataModel):
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return

    # Variables
    # demand_supply[demand_point_index, supply_point_index] = proportion of demand at
    # demand_index satisfied by supply at supply_index
    demand_supply = {}
    for demand_point_index in data_model.demand_point_indexes:
        for supply_point_index in data_model.supply_point_indexes:
            demand_supply[(demand_point_index, supply_point_index)] = solver.IntVar(
                0, 100, f"demand_supply_{demand_point_index}_{supply_point_index}"
            )

    supply_point_index_capacities = {}
    for charging_stations in [SLOW_CHARGING_STATION_TAG, FAST_CHARGING_STATION_TAG]:
        for supply_point_index in data_model.supply_point_indexes:
            supply_point_index_capacities[(charging_stations, supply_point_index)] = solver.IntVar(
                data_model.supply_point_index_low_capacities[charging_stations, supply_point_index],
                (
                    data_model.supply_point_index_low_capacities[charging_stations, supply_point_index]
                    + MAX_INCREASING_CAPACITIES
                ),
                f"supply_point_index_capacities_{charging_stations}_{supply_point_index}",
            )

    # Constraints
    # All values of the demand_supply matrix (DS_{i, j}) must be non-negative. (deja remplie ...)
    # for demand_point_index in data_model.demand_point_indexes:
    #     for supply_point_index in data_model.supply_point_indexes:
    #         solver.Add(demand_supply[demand_point_index, supply_point_index] >= 0)

    # All values of the number of slow (SCS_j) and fast charging stations (FCS_j) must be a
    # non-negative integer. (deja remplie ...)

    # Sum of slow (SCS_j) and fast charging stations (FCS_j) must be less than or equal to
    # the total parking slots (PS_j) available at each jth supply point
    for supply_point_index in data_model.supply_point_indexes:
        solver.Add(
            sum(
                supply_point_index_capacities[charging_stations, supply_point_index]
                for charging_stations in [SLOW_CHARGING_STATION_TAG, FAST_CHARGING_STATION_TAG]
            )
            <= data_model.total_parking_slots[supply_point_index]
        )

    # TODO: To implement when dealing with several years
    # You can only build incremental EV infrastructure on top of the 2018 infrastructure.
    # That means, SCS_j and FCS_j must increase or stay constant year-on-year at each 𝑗𝑗𝑡𝑡ℎ
    # supply point.

    # (Sum of fractional) Demand satisfied by each jth supply point must be less than or
    # equal to the maximum supply available.
    for supply_point_index in data_model.supply_point_indexes:
        solver.Add(
            sum(
                demand_supply[(demand_point_index, supply_point_index)]
                for demand_point_index in data_model.demand_point_indexes
            )
            <= MAXIMUM_SUPPLY_PROPORTION_AVAILABLE
        )

    # (Sum of fractional) Forecasted demand at each ith demand point must exactly be satisfied.
    for demand_point_index in data_model.demand_point_indexes:
        solver.Add(
            sum(
                demand_supply[(demand_point_index, supply_point_index)]
                for supply_point_index in data_model.supply_point_indexes
            )
            == data_model.demand_forecast[demand_point_index]
        )

    # TODO: Objective: minimize the number of bins used.
    solver.Minimize(solver.Sum())

    return None


if __name__ == "__main__":
    logging.info("Start Main")
    data_path = Path(__file__).parent.parent / "data"

    demand_history: pd.DataFrame = pd.read_csv(data_path / "Demand_History.csv")
    existing_infrastructure: pd.DataFrame = pd.read_csv(data_path / "exisiting_EV_infrastructure_2018.csv")

    data_model = create_data_model(demand_history, existing_infrastructure)

    main(data_model)
