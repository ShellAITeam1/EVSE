from typing import Dict, List

from ortools.linear_solver.linear_solver_natural_api import ProductCst

from ..const import (
    CUSTOMER_DISSATISFACTION_LOSS_COEFFICIENT,
    FAST_CHARGING_COST_RATIO,
    FAST_CHARGING_STATION_TAG,
    INFRASTRUCTURES_GROWTH_LOSS_COEFFICIENT,
    SLOW_CHARGING_STATION_TAG,
)


def compute_customer_dissatisfaction(
    distance_matrix: Dict, demand_supply: Dict, demand_forecast: Dict, year_list: List[int]
) -> List[ProductCst]:
    distance_by_demand = [
        CUSTOMER_DISSATISFACTION_LOSS_COEFFICIENT
        * distance_matrix[index_couple]
        * demand_supply[year, index_couple[0], index_couple[1]]
        * demand_forecast[year, index_couple[0]]
        for index_couple in distance_matrix.keys()
        for year in year_list
    ]

    return distance_by_demand


def compute_cost_of_infrastructure(
    supply_point_index_capacities: Dict, years_list: List[int], supply_point_indexes: List[int]
) -> List[ProductCst]:
    total_capacities = []
    for year in years_list:
        for supply_point_index in supply_point_indexes:
            point_total_capacity = INFRASTRUCTURES_GROWTH_LOSS_COEFFICIENT * (
                supply_point_index_capacities[year, SLOW_CHARGING_STATION_TAG, supply_point_index]
                + FAST_CHARGING_COST_RATIO
                * supply_point_index_capacities[year, FAST_CHARGING_STATION_TAG, supply_point_index]
            )
            total_capacities.append(point_total_capacity)

    return total_capacities
