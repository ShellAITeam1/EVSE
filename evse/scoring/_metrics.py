from typing import List

import numpy as np

from evse.const import (
    CUSTOMER_DISSATISFACTION_LOSS_COEFFICIENT,
    DEMAND_MISMATCH_LOSS_COEFFICIENT,
    FAST_CHARGING_COST_RATIO,
    INFRASTRUCTURES_GROWTH_LOSS_COEFFICIENT,
)
from evse.scoring._submission import Result


def compute_customer_dissatisfaction_cost_over_years(
    distance_matrix: np.array, demand_supply_values_list: List[np.array]
) -> float:
    return np.sum(distance_matrix * sum(demand_supply_values_list))


def compute_demand_mismatch_cost(demand_true: np.array, demand_pred: np.array) -> float:
    return np.sum(np.abs(demand_true - demand_pred))


def compute_demand_mismatch_cost_over_years(
    demand_true_list: List[np.array], demand_pred_list: List[np.array]
) -> float:
    demand_mismatch_cost_over_years = sum(
        [
            compute_demand_mismatch_cost(demand_true, demand_pred)
            for demand_true, demand_pred in zip(demand_true_list, demand_pred_list)
        ]
    )
    return demand_mismatch_cost_over_years


def compute_cost_of_infrastructures_on_one_year(
    slow_charging_stations_on_supply_point_matrix: np.array,
    fast_charging_stations_on_supply_point_matrix: np.array,
) -> float:
    cost_of_infrastructures_on_one_year = np.sum(
        slow_charging_stations_on_supply_point_matrix
        + FAST_CHARGING_COST_RATIO * fast_charging_stations_on_supply_point_matrix
    )
    return cost_of_infrastructures_on_one_year


def compute_cost_of_infrastructures_over_years(
    slow_charging_stations_on_supply_point_matrix_list: List[np.array],
    fast_charging_stations_on_supply_point_matrix_list: List[np.array],
) -> float:
    all_year_compute_cost_of_infrastructures = []
    for (slow_charging_stations_on_supply_point_matrix, fast_charging_stations_on_supply_point_matrix,) in zip(
        slow_charging_stations_on_supply_point_matrix_list,
        fast_charging_stations_on_supply_point_matrix_list,
    ):
        yearly_cost = compute_cost_of_infrastructures_on_one_year(
            slow_charging_stations_on_supply_point_matrix,
            fast_charging_stations_on_supply_point_matrix,
        )
        all_year_compute_cost_of_infrastructures.append(yearly_cost)
    return sum(all_year_compute_cost_of_infrastructures)


def score_loss(yearly_results_list: List[Result], distance_matrix: np.array):
    customer_dissatisfaction = compute_customer_dissatisfaction_cost_over_years(
        distance_matrix,
        [yearly_result.demand_supply_matrix.values for yearly_result in yearly_results_list],
    )

    demand_mismatch_cost = 0

    cost_of_infrastructures = compute_cost_of_infrastructures_over_years(
        [
            yearly_result.slow_charging_stations_on_supply_point_matrix["value"].values
            for yearly_result in yearly_results_list
        ],
        [
            yearly_result.fast_charging_stations_on_supply_point_matrix["value"].values
            for yearly_result in yearly_results_list
        ],
    )

    return max(
        10,
        100
        - 90
        * (
            CUSTOMER_DISSATISFACTION_LOSS_COEFFICIENT * customer_dissatisfaction
            + DEMAND_MISMATCH_LOSS_COEFFICIENT * demand_mismatch_cost
            + INFRASTRUCTURES_GROWTH_LOSS_COEFFICIENT * cost_of_infrastructures
        )
        / 15000000,
    )
