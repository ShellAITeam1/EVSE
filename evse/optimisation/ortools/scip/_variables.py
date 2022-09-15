from typing import Dict

from ortools.linear_solver import pywraplp

from evse.const import FAST_CHARGING_STATION_TAG, SLOW_CHARGING_STATION_TAG
from evse.optimisation import DataModel


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
                    0.00000001, 1, f"demand_supply_{year}_{demand_point_index}_{supply_point_index}"
                )
    return demand_supply


MAX_INCREASING_CAPACITIES = 100
