from typing import Dict

from ortools.linear_solver import pywraplp

from evse.const import FAST_CHARGING_STATION_TAG, SLOW_CHARGING_STATION_TAG
from evse.optimisation import DataModel


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
                - 0.0001
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
