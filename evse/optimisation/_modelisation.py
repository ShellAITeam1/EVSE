import pickle
from pathlib import Path
from typing import Dict, List, Protocol

import numpy as np
import pandas as pd
from tqdm import tqdm

from evse.const import (
    CAP_FCS,
    CAP_SCS,
    DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
    DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
    DEMAND_POINT_INDEX_COLUMN_NAME,
    EXISTING_EV_TOTAL_PARKING_SLOT_COLUMN_NAME,
    FAST_CHARGING_STATION_TAG,
    SLOW_CHARGING_STATION_TAG,
    SUPPLY_POINT_INDEX_COLUMN_NAME,
)


class DataModel(Protocol):
    supply_point_indexes: List[int]
    demand_point_indexes: List[int]
    supply_point_index_low_capacities: Dict
    demand_forecast: Dict
    total_parking_slots: Dict
    distance_matrix: Dict
    scs_capacity: int
    fcs_capacity: int
    years_list: List[int]


class EVSEDataModel:
    def __init__(
        self,
        supply_point_indexes: List[int],
        demand_point_indexes: List[int],
        supply_point_index_low_capacities: Dict,
        demand_forecast: Dict,
        total_parking_slots: Dict,
        distance_matrix: Dict,
        scs_capacity: int,
        fcs_capacity: int,
        years_list: List[int],
    ):
        self.supply_point_indexes = supply_point_indexes
        self.demand_point_indexes = demand_point_indexes
        self.supply_point_index_low_capacities = supply_point_index_low_capacities
        self.demand_forecast = demand_forecast
        self.total_parking_slots = total_parking_slots
        self.distance_matrix = distance_matrix
        self.scs_capacity = scs_capacity
        self.fcs_capacity = fcs_capacity
        self.years_list = years_list


def extract_distance_matrix(
    demand_history_df: pd.DataFrame, existing_infrastructure_df: pd.DataFrame, saving_path
) -> Dict:
    distance_matrix_path = saving_path / "distance_matrix.pickle"
    if distance_matrix_path.is_file():
        with open(distance_matrix_path, "rb") as handle:
            distance_matrix = pickle.load(handle)
    else:
        distance_matrix = {}
        for demand_point_index in tqdm(demand_history_df[DEMAND_POINT_INDEX_COLUMN_NAME].unique()):
            for supply_point_index in existing_infrastructure_df[SUPPLY_POINT_INDEX_COLUMN_NAME].unique():
                current_demand_coordinate = demand_history_df[
                    demand_history_df[DEMAND_POINT_INDEX_COLUMN_NAME] == demand_point_index
                ][[DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME, DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME]].values
                current_supply_coordinate = existing_infrastructure_df[
                    existing_infrastructure_df[SUPPLY_POINT_INDEX_COLUMN_NAME] == supply_point_index
                ][[DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME, DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME]].values
                distance_matrix[demand_point_index, supply_point_index] = np.sqrt(
                    np.sum((current_demand_coordinate - current_supply_coordinate) ** 2)
                )

        with open(distance_matrix_path, "wb") as f:
            pickle.dump(distance_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

    return distance_matrix


def extract_years_list(demand_forecast_df: pd.DataFrame) -> List[int]:
    extract_years_list = demand_forecast_df.columns.difference([DEMAND_POINT_INDEX_COLUMN_NAME]).astype(int).to_list()
    return extract_years_list


def create_data_model(
    demand_forecast_df: pd.DataFrame,
    demand_history_df: pd.DataFrame,
    existing_infrastructure_df: pd.DataFrame,
    saving_path: Path,
) -> DataModel:

    supply_point_index_lower_capacities = extract_supply_point_index_lower_capacities(existing_infrastructure_df)

    years_list = extract_years_list(demand_forecast_df)

    demand_forecast = extract_demand_forecast(demand_forecast_df, years_list)

    total_parking_slots = extract_total_parking_slots(existing_infrastructure_df)

    distance_matrix = extract_distance_matrix(demand_history_df, existing_infrastructure_df, saving_path)

    data = EVSEDataModel(
        existing_infrastructure_df[SUPPLY_POINT_INDEX_COLUMN_NAME].unique().tolist(),
        demand_history_df[DEMAND_POINT_INDEX_COLUMN_NAME].unique().tolist(),
        supply_point_index_lower_capacities,
        demand_forecast,
        total_parking_slots,
        distance_matrix,
        CAP_SCS,
        CAP_FCS,
        years_list,
    )
    return data


def extract_total_parking_slots(existing_infrastructure: pd.DataFrame) -> Dict:
    return {
        supply_point_index: existing_infrastructure[
            existing_infrastructure[SUPPLY_POINT_INDEX_COLUMN_NAME] == supply_point_index
        ][EXISTING_EV_TOTAL_PARKING_SLOT_COLUMN_NAME].iloc[0]
        for supply_point_index in existing_infrastructure[SUPPLY_POINT_INDEX_COLUMN_NAME].unique()
    }


def extract_demand_forecast(demand_history_df: pd.DataFrame, years_list: List[int]) -> Dict:
    return {
        (year, demand_point_index): demand_history_df[
            demand_history_df[DEMAND_POINT_INDEX_COLUMN_NAME] == demand_point_index
        ][f"{year}"].iloc[0]
        for demand_point_index in demand_history_df[DEMAND_POINT_INDEX_COLUMN_NAME].unique()
        for year in years_list
    }


def extract_supply_point_index_lower_capacities(existing_infrastructure: pd.DataFrame) -> Dict:
    mapping_scs_fcs_columns = {
        SLOW_CHARGING_STATION_TAG: "existing_num_SCS",
        FAST_CHARGING_STATION_TAG: "existing_num_FCS",
    }
    return {
        (charging_stations, supply_point_index): existing_infrastructure[
            existing_infrastructure[SUPPLY_POINT_INDEX_COLUMN_NAME] == supply_point_index
        ][mapping_scs_fcs_columns[charging_stations]].iloc[0]
        for charging_stations in [SLOW_CHARGING_STATION_TAG, FAST_CHARGING_STATION_TAG]
        for supply_point_index in existing_infrastructure[SUPPLY_POINT_INDEX_COLUMN_NAME].unique()
    }
