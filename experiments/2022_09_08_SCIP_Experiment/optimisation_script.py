"""Main module."""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from ortools.linear_solver import pywraplp

from evse.const import (
    DEMAND_POINT_INDEX_COLUMN_NAME,
    FAST_CHARGING_STATION_TAG,
    SLOW_CHARGING_STATION_TAG,
)
from evse.logger import logger
from evse.optimisation import (
    DataModel,
    SolverInitiationError,
    compute_all_years_results,
    compute_cost_of_infrastructure,
    compute_customer_dissatisfaction,
    create_data_model,
)
from evse.optimisation._result import summarize_optimization_output
from evse.scoring import Result

MAX_INCREASING_CAPACITIES = 100


def main(data_model: DataModel, limit_in_time: Optional[int] = None, show_output: bool = False) -> List[Result]:
    # Create the mip solver with the SCIP backend.
    logger.info("Create the mip solver with the SCIP backend")
    solver = pywraplp.Solver("my_model", pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
    if limit_in_time:
        logger.info(f"Time limit set to {limit_in_time} seconds.")
        solver.set_time_limit(limit_in_time)

    if not solver:
        raise SolverInitiationError("Solver not created")

    # Variables
    logger.info("Initializing Variables")
    logger.info("Initializing Variables: demand_supply")
    demand_supply = create_demand_supply_variable(data_model, solver)

    logger.info("Initializing Variables: supply_point_index_capacities")
    supply_point_index_capacities = create_supply_point_index_capacities_variable(data_model, solver)

    # Constraints
    logger.info("Setting Constraints")
    add_demand_supply_positive_or_null_constraint()
    add_number_of_charging_station_is_non_null_positive_integer()
    add_parking_slot_per_supply_point_index_constraint(data_model, solver, supply_point_index_capacities)
    add_incremental_ev_infrastructure_constraint(data_model, solver, supply_point_index_capacities)
    add_demand_point_delivery_should_be_less_than_capacity_constraint(
        data_model, demand_supply, solver, supply_point_index_capacities
    )
    add_demand_point_partition_over_all_supply_point_constraint(data_model, demand_supply, solver)

    logger.info("Setting Objective Function")
    solver.Minimize(
        solver.Sum(
            [
                solver.Sum(
                    compute_customer_dissatisfaction(
                        data_model.distance_matrix, demand_supply, data_model.demand_forecast, data_model.years_list
                    )
                ),
                solver.Sum(
                    compute_cost_of_infrastructure(
                        supply_point_index_capacities, data_model.years_list, data_model.supply_point_indexes
                    )
                ),
            ]
        )
    )

    solver.EnableOutput()
    logger.info("Start Solving")
    status = solver.Solve()

    all_years_results = compute_all_years_results(status, demand_supply, supply_point_index_capacities, data_model)

    if show_output and len(all_years_results) > 0:
        summarize_optimization_output(all_years_results, data_model)

    return all_years_results


def add_demand_point_partition_over_all_supply_point_constraint(
    data_model: DataModel, demand_supply: Dict, solver: pywraplp.Solver
) -> None:
    # (Sum of fractional) Forecasted demand at each ith demand point must exactly be satisfied.
    for year in data_model.years_list:
        for demand_point_index in data_model.demand_point_indexes:
            solver.Add(
                sum(
                    demand_supply[(year, demand_point_index, supply_point_index)]
                    for supply_point_index in data_model.supply_point_indexes
                )
                == 1
            )


def add_demand_point_delivery_should_be_less_than_capacity_constraint(
    data_model: DataModel, demand_supply: Dict, solver: pywraplp.Solver, supply_point_index_capacities: Dict
) -> None:
    # (Sum of fractional) Demand satisfied by each jth supply point must be less than or
    # equal to the maximum supply available.
    for year in data_model.years_list:
        for supply_point_index in data_model.supply_point_indexes:
            solver.Add(
                (
                    data_model.scs_capacity
                    * supply_point_index_capacities[year, SLOW_CHARGING_STATION_TAG, supply_point_index]
                    + data_model.fcs_capacity
                    * supply_point_index_capacities[year, FAST_CHARGING_STATION_TAG, supply_point_index]
                )
                >= sum(
                    demand_supply[(year, demand_point_index, supply_point_index)]
                    * data_model.demand_forecast[year, demand_point_index]
                    for demand_point_index in data_model.demand_point_indexes
                )
            )


def add_number_of_charging_station_is_non_null_positive_integer() -> None:
    # All values of the number of slow (SCS_j) and fast charging stations (FCS_j) must be a
    # non-negative integer. (deja remplie ...)
    pass


def add_demand_supply_positive_or_null_constraint() -> None:
    # All values of the demand_supply matrix (DS_{i, j}) must be non-negative. (deja remplie ...)
    pass


def add_incremental_ev_infrastructure_constraint(
    data_model: DataModel, solver: pywraplp.Solver, supply_point_index_capacities: Dict
) -> None:
    # You can only build incremental EV infrastructure on top of the 2018 infrastructure.
    # That means, SCS_j and FCS_j must increase or stay constant year-on-year at each jth
    # supply point.
    for year in data_model.years_list[1:]:
        for charging_stations in [SLOW_CHARGING_STATION_TAG, FAST_CHARGING_STATION_TAG]:
            for supply_point_index in data_model.supply_point_indexes:
                solver.Add(
                    (
                        supply_point_index_capacities[(year - 1, charging_stations, supply_point_index)]
                        <= supply_point_index_capacities[(year, charging_stations, supply_point_index)]
                    )
                )


def add_parking_slot_per_supply_point_index_constraint(
    data_model: DataModel, solver: pywraplp.Solver, supply_point_index_capacities: Dict
) -> None:
    # Sum of slow (SCS_j) and fast charging stations (FCS_j) must be less than or equal to
    # the total parking slots (PS_j) available at each jth supply point
    for year in data_model.years_list:
        for supply_point_index in data_model.supply_point_indexes:
            solver.Add(
                sum(
                    supply_point_index_capacities[year, charging_stations, supply_point_index]
                    for charging_stations in [SLOW_CHARGING_STATION_TAG, FAST_CHARGING_STATION_TAG]
                )
                <= data_model.total_parking_slots[supply_point_index]
            )


def create_supply_point_index_capacities_variable(data_model: DataModel, solver: pywraplp.Solver) -> Dict:
    supply_point_index_capacities = {}
    for year in data_model.years_list:
        for charging_stations in [SLOW_CHARGING_STATION_TAG, FAST_CHARGING_STATION_TAG]:
            for supply_point_index in data_model.supply_point_indexes:
                supply_point_index_capacities[(year, charging_stations, supply_point_index)] = solver.IntVar(
                    int(data_model.supply_point_index_low_capacities[charging_stations, supply_point_index]),
                    int(
                        data_model.supply_point_index_low_capacities[charging_stations, supply_point_index]
                        + MAX_INCREASING_CAPACITIES
                    ),
                    f"supply_point_index_capacities_{year}_{charging_stations}_{supply_point_index}",
                )
    return supply_point_index_capacities


def create_demand_supply_variable(data_model: DataModel, solver: pywraplp.Solver) -> Dict:
    # demand_supply[demand_point_index, supply_point_index] = proportion of demand at
    # demand_index satisfied by supply at supply_index
    demand_supply = {}
    for year in data_model.years_list:
        for demand_point_index in data_model.demand_point_indexes:
            for supply_point_index in data_model.supply_point_indexes:
                demand_supply[(year, demand_point_index, supply_point_index)] = solver.NumVar(
                    0, 1, f"demand_supply_{year}_{demand_point_index}_{supply_point_index}"
                )
    return demand_supply


if __name__ == "__main__":
    logger.info("Start Main")
    data_path = Path(__file__).parent.parent.parent / "data"

    demand_history: pd.DataFrame = pd.read_csv(data_path / "Demand_History.csv")
    demand_forecast: pd.DataFrame = pd.read_csv(data_path / "Demand_History.csv")
    demand_forecast["2019"] = demand_forecast["2018"] * 1.1
    demand_forecast["2020"] = demand_forecast["2019"] * 1.15
    demand_forecast = demand_forecast[[DEMAND_POINT_INDEX_COLUMN_NAME, "2019", "2020"]]
    existing_infrastructure: pd.DataFrame = pd.read_csv(data_path / "exisiting_EV_infrastructure_2018.csv")

    # restricted_demand_point = range(1000)#np.random.choice(demand_forecast[DEMAND_POINT_INDEX_COLUMN_NAME], 400)
    # restricted_supply_point = range(20)# np.random.choice(existing_infrastructure[SUPPLY_POINT_INDEX_COLUMN_NAME], 10)
    # demand_forecast = demand_forecast[
    #     demand_forecast[DEMAND_POINT_INDEX_COLUMN_NAME].isin(restricted_demand_point)
    # ]
    # demand_history = demand_history[
    #     demand_history[DEMAND_POINT_INDEX_COLUMN_NAME].isin(restricted_demand_point)
    # ]
    # existing_infrastructure = existing_infrastructure[
    #     existing_infrastructure[SUPPLY_POINT_INDEX_COLUMN_NAME].isin(restricted_supply_point)
    # ]

    data_model = create_data_model(demand_forecast, demand_history, existing_infrastructure, data_path)

    # data_model.distance_matrix = {
    #     key: value for key, value in data_model.distance_matrix.items()
    #     if key[0] in restricted_demand_point and key[1] in restricted_supply_point
    # }

    main(data_model, show_output=True)
