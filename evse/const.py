SUPPLY_POINT_INDEX_COLUMN_NAME = "supply_point_index"
DEMAND_POINT_INDEX_COLUMN_NAME = "demand_point_index"
YEAR_INDEX_COLUMN_NAME = "year"
DATA_TYPE_COLUMN_NAME = "data_type"
VALUE_COLUMN_NAME = "value"

ORDERED_SUBMISSION_DATAFRAME_COLUMNS_NAME_LIST = [
    YEAR_INDEX_COLUMN_NAME,
    DATA_TYPE_COLUMN_NAME,
    DEMAND_POINT_INDEX_COLUMN_NAME,
    SUPPLY_POINT_INDEX_COLUMN_NAME,
    VALUE_COLUMN_NAME,
]

SUBMISSION_DATAFRAME_TYPE_DICT = {
    YEAR_INDEX_COLUMN_NAME: int,
    DATA_TYPE_COLUMN_NAME: str,
    DEMAND_POINT_INDEX_COLUMN_NAME: float,
    SUPPLY_POINT_INDEX_COLUMN_NAME: int,
    VALUE_COLUMN_NAME: float,
}

SCS_VALUE_COLUMN_NAME = "scs_value"
FCS_VALUE_COLUMN_NAME = "fcs_value"

NUMBER_OF_SUPPLY_POINTS = 100
NUMBER_OF_DEMAND_POINT = 4096

SLOW_CHARGING_STATION_TAG = "SCS"
FAST_CHARGING_STATION_TAG = "FCS"

EXISTING_EV_TOTAL_PARKING_SLOT_COLUMN_NAME = "total_parking_slots"
EXISTING_EV_X_COORDINATE_COLUMN_NAME = "x_coordinate"
EXISTING_EV_Y_COORDINATE_COLUMN_NAME = "y_coordinate"

DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME = "x_coordinate"
DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME = "y_coordinate"

FAST_CHARGING_COST_RATIO = 1.5
CAP_SCS = 200
CAP_FCS = 400

CUSTOMER_DISSATISFACTION_LOSS_COEFFICIENT = 1
DEMAND_MISMATCH_LOSS_COEFFICIENT = 25
INFRASTRUCTURES_GROWTH_LOSS_COEFFICIENT = 600


# Forecasters
RFR = {"cv": True, "name": "RandomForestRegressor"}
SPL = {"cv": False, "name": "Spline"}
SPD = {"cv": False, "name": "SplinePerDemand"}



