from typing import Dict, List

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from evse.const import (
    DATA_TYPE_COLUMN_NAME,
    DEMAND_POINT_INDEX_COLUMN_NAME,
    FAST_CHARGING_STATION_TAG,
    SLOW_CHARGING_STATION_TAG,
    SUPPLY_POINT_INDEX_COLUMN_NAME,
    VALUE_COLUMN_NAME,
)
from evse.logger import logger
from evse.optimisation import DataModel
from evse.scoring import Result, YearlyResult, score_loss


def compute_all_years_results(
    status: int,
    demand_supply: Dict,
    supply_point_index_capacities: Dict,
    data_model: DataModel,
) -> List[Result]:
    all_years_results: List[Result] = []
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.AT_LOWER_BOUND:
        for year in data_model.years_list:
            demand_supply_solution_values = np.array(
                [
                    [
                        demand_supply[year, demand_point_index, supply_point_index].solution_value()
                        * data_model.demand_forecast[year, demand_point_index]
                        for supply_point_index in data_model.supply_point_indexes
                    ]
                    for demand_point_index in data_model.demand_point_indexes
                ]
            )

            demand_supply_solution_columns = pd.MultiIndex.from_tuples(
                [(VALUE_COLUMN_NAME, supply_point_index) for supply_point_index in data_model.supply_point_indexes],
                names=[None, SUPPLY_POINT_INDEX_COLUMN_NAME],
            )

            demand_supply_solution_df = pd.DataFrame(
                demand_supply_solution_values,
                columns=demand_supply_solution_columns,
                index=pd.Index(data_model.demand_point_indexes, name=DEMAND_POINT_INDEX_COLUMN_NAME),
            )

            supply_point_index_capacities_solution_df = pd.DataFrame(
                [
                    {
                        DATA_TYPE_COLUMN_NAME: point_supply_capacities_index[1],
                        SUPPLY_POINT_INDEX_COLUMN_NAME: point_supply_capacities_index[2],
                        VALUE_COLUMN_NAME: point_supply_capacities.solution_value(),
                    }
                    for point_supply_capacities_index, point_supply_capacities in supply_point_index_capacities.items()
                    if point_supply_capacities_index[0] == year
                ]
            )

            scs_supply_point_index_capacities_solution_df = supply_point_index_capacities_solution_df[
                supply_point_index_capacities_solution_df[DATA_TYPE_COLUMN_NAME] == SLOW_CHARGING_STATION_TAG
            ][[SUPPLY_POINT_INDEX_COLUMN_NAME, VALUE_COLUMN_NAME]].reset_index(drop=True)

            fcs_supply_point_index_capacities_solution_df = supply_point_index_capacities_solution_df[
                supply_point_index_capacities_solution_df[DATA_TYPE_COLUMN_NAME] == FAST_CHARGING_STATION_TAG
            ][[SUPPLY_POINT_INDEX_COLUMN_NAME, VALUE_COLUMN_NAME]].reset_index(drop=True)

            yearly_result = YearlyResult(
                year=year,
                slow_charging_stations_on_supply_point_matrix=scs_supply_point_index_capacities_solution_df,
                fast_charging_stations_on_supply_point_matrix=fcs_supply_point_index_capacities_solution_df,
                demand_supply_matrix=demand_supply_solution_df,
            )

            all_years_results.append(yearly_result)

    else:
        logger.info("The problem does not have an optimal solution.")
    return all_years_results


def summarize_optimization_output(all_years_results: List[Result], data_model: DataModel) -> None:
    logger.info("Optimization Results:")
    distance_matrix_in_array = np.ones((len(data_model.demand_point_indexes), len(data_model.supply_point_indexes)))
    for index_couple, distance in data_model.distance_matrix.items():
        distance_matrix_in_array[index_couple] = distance

    optimization_score = score_loss(all_years_results, distance_matrix_in_array)

    logger.info(f"Final Score: {optimization_score}\n\n")
    for supply_point_index in data_model.supply_point_indexes:
        scs_values = [data_model.supply_point_index_low_capacities[SLOW_CHARGING_STATION_TAG, supply_point_index]]
        fcs_values = [data_model.supply_point_index_low_capacities[FAST_CHARGING_STATION_TAG, supply_point_index]]

        for year_result in all_years_results:
            yearly_scs_value = year_result.slow_charging_stations_on_supply_point_matrix.loc[
                (
                    year_result.slow_charging_stations_on_supply_point_matrix[SUPPLY_POINT_INDEX_COLUMN_NAME]
                    == supply_point_index
                ),
                "value",
            ].values[0]
            scs_values.append(yearly_scs_value)
            yearly_fcs_value = year_result.fast_charging_stations_on_supply_point_matrix.loc[
                (
                    year_result.fast_charging_stations_on_supply_point_matrix[SUPPLY_POINT_INDEX_COLUMN_NAME]
                    == supply_point_index
                ),
                "value",
            ].values[0]
            fcs_values.append(yearly_fcs_value)

        str_convertor = lambda x: str(int(x))
        scs_evolution_string = " -> ".join(map(str_convertor, scs_values))
        fcs_evolution_string = " -> ".join(map(str_convertor, fcs_values))
        logger.info(
            f"Supply point index: {str(supply_point_index)}:"
            f"{SLOW_CHARGING_STATION_TAG}-[{scs_evolution_string}]  "
            f"{FAST_CHARGING_STATION_TAG}-[{fcs_evolution_string}]  ",
        )
