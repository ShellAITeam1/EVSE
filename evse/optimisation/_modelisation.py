from typing import Dict, List, Protocol

import pandas as pd

from evse.const import (
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


class EVSEDataModel:
    def __init__(
        self,
        supply_point_indexes: List[int],
        demand_point_indexes: List[int],
        supply_point_index_low_capacities: Dict,
        demand_forecast: Dict,
        total_parking_slots: Dict,
    ):
        self.supply_point_indexes = supply_point_indexes
        self.demand_point_indexes = demand_point_indexes
        self.supply_point_index_low_capacities = supply_point_index_low_capacities
        self.demand_forecast = demand_forecast
        self.total_parking_slots = total_parking_slots


def create_data_model(demand_history: pd.DataFrame, existing_infrastructure: pd.DataFrame) -> DataModel:

    supply_point_index_lower_capacities = extract_supply_point_index_lower_capacities(existing_infrastructure)

    demand_forecast = extract_demand_forecast(demand_history)

    total_parking_slots = extract_total_parking_slots(existing_infrastructure)

    data = EVSEDataModel(
        demand_history[DEMAND_POINT_INDEX_COLUMN_NAME].unique().tolist(),
        demand_history[SUPPLY_POINT_INDEX_COLUMN_NAME].unique().tolist(),
        supply_point_index_lower_capacities,
        demand_forecast,
        total_parking_slots,
    )
    return data


def extract_total_parking_slots(existing_infrastructure: pd.DataFrame) -> Dict:
    return {
        supply_point_index: existing_infrastructure[
            existing_infrastructure[SUPPLY_POINT_INDEX_COLUMN_NAME] == supply_point_index
        ][EXISTING_EV_TOTAL_PARKING_SLOT_COLUMN_NAME].iloc[0]
        for supply_point_index in existing_infrastructure[SUPPLY_POINT_INDEX_COLUMN_NAME].unique()
    }


def extract_demand_forecast(demand_history: pd.DataFrame) -> Dict:
    return {
        demand_point_index: demand_history[demand_history[DEMAND_POINT_INDEX_COLUMN_NAME] == demand_point_index][
            "2018"
        ].iloc[0]
        for demand_point_index in demand_history[DEMAND_POINT_INDEX_COLUMN_NAME].unique()
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
