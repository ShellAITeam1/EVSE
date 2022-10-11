from typing import List, Protocol

import pandas as pd

from evse.const import (
    DATA_TYPE_COLUMN_NAME,
    DEMAND_POINT_INDEX_COLUMN_NAME,
    ORDERED_SUBMISSION_DATAFRAME_COLUMNS_NAME_LIST,
    SUBMISSION_DATAFRAME_TYPE_DICT,
    SUPPLY_POINT_INDEX_COLUMN_NAME,
    YEAR_INDEX_COLUMN_NAME,
)


class Result(Protocol):
    year: int
    slow_charging_stations_on_supply_point_matrix: pd.DataFrame
    fast_charging_stations_on_supply_point_matrix: pd.DataFrame
    demand_supply_matrix: pd.DataFrame
    slow_charging_stations_on_supply_point_matrix_data_type: str
    fast_charging_stations_on_supply_point_matrix_data_type: str
    demand_supply_matrix_data_type: str

    def build_yearly_submission_dataframe(self) -> pd.DataFrame:
        pass


class YearlyResult:
    slow_charging_stations_on_supply_point_matrix_data_type = "SCS"
    fast_charging_stations_on_supply_point_matrix_data_type = "FCS"
    demand_supply_matrix_data_type = "DS"

    def __init__(
        self,
        year: int,
        slow_charging_stations_on_supply_point_matrix: pd.DataFrame,
        fast_charging_stations_on_supply_point_matrix: pd.DataFrame,
        demand_supply_matrix: pd.DataFrame,
    ):
        self.year = year
        self.slow_charging_stations_on_supply_point_matrix = slow_charging_stations_on_supply_point_matrix
        self.fast_charging_stations_on_supply_point_matrix = fast_charging_stations_on_supply_point_matrix
        self.demand_supply_matrix = demand_supply_matrix

    def __repr__(self):
        return f"{self.__class__.__name__}(year={self.year})"

    def build_yearly_submission_dataframe(self) -> pd.DataFrame:
        demand_supply_matrix_stacked = self.demand_supply_matrix.stack(SUPPLY_POINT_INDEX_COLUMN_NAME).reset_index()
        demand_supply_matrix_stacked[YEAR_INDEX_COLUMN_NAME] = self.year
        demand_supply_matrix_stacked[DATA_TYPE_COLUMN_NAME] = self.demand_supply_matrix_data_type

        slow_charging_stations_on_supply_point_matrix_stacked = (
            self.slow_charging_stations_on_supply_point_matrix.copy()
        )
        slow_charging_stations_on_supply_point_matrix_stacked[YEAR_INDEX_COLUMN_NAME] = self.year
        slow_charging_stations_on_supply_point_matrix_stacked[
            DATA_TYPE_COLUMN_NAME
        ] = self.slow_charging_stations_on_supply_point_matrix_data_type

        fast_charging_stations_on_supply_point_matrix_stacked = (
            self.fast_charging_stations_on_supply_point_matrix.copy()
        )
        fast_charging_stations_on_supply_point_matrix_stacked[YEAR_INDEX_COLUMN_NAME] = self.year
        fast_charging_stations_on_supply_point_matrix_stacked[
            DATA_TYPE_COLUMN_NAME
        ] = self.fast_charging_stations_on_supply_point_matrix_data_type

        yearly_submission_dataframe = pd.concat(
            [
                slow_charging_stations_on_supply_point_matrix_stacked,
                fast_charging_stations_on_supply_point_matrix_stacked,
                demand_supply_matrix_stacked,
            ],
            axis=0,
        )

        yearly_submission_dataframe = yearly_submission_dataframe.infer_objects()

        return yearly_submission_dataframe


def get_scoring_dataframe(yearly_results: List[Result]) -> pd.DataFrame:
    all_years_df = pd.concat(
        [yearly_result.build_yearly_submission_dataframe() for yearly_result in yearly_results],
        axis=0,
    )
    all_years_df = all_years_df[ORDERED_SUBMISSION_DATAFRAME_COLUMNS_NAME_LIST]
    all_years_df = (
        all_years_df.astype(SUBMISSION_DATAFRAME_TYPE_DICT)
        .set_index(
            [
                YEAR_INDEX_COLUMN_NAME,
                DATA_TYPE_COLUMN_NAME,
                DEMAND_POINT_INDEX_COLUMN_NAME,
                SUPPLY_POINT_INDEX_COLUMN_NAME,
            ]
        )
        .sort_index(ascending=[True, False, True, True])
        .reset_index()
    )

    return all_years_df
