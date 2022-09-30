from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

import evse
from evse.const import (
    DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
    DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
    DEMAND_POINT_INDEX_COLUMN_NAME,
    VALUE_COLUMN_NAME,
    YEAR_INDEX_COLUMN_NAME,
)
from evse.forecast import ForecasterTrainingResult, TrainingResult
from evse.optimisation import create_data_model
from evse.optimisation.ortools.scip import apply_scip_optimizer
from evse.scoring import get_scoring_dataframe


def get_neighbour(index, radius):
    x_lower_bound = index[1] - radius
    y_lower_bound = index[2] - radius

    filtered_neighbour_indexes = [
        (demand_point, x_lower_bound + x, y_lower_bound + y, index[3])
        for demand_point in range(
            index[0] - 2 * radius * 64, index[0] + 2 * radius * 64
        )
        for x in range(2 * radius + 1)
        for y in range(2 * radius + 1)
        if (index[0], x, y, index[3]) != index
    ]
    return filtered_neighbour_indexes


def compute_neighbour(index, dataframe, radius):
    neighbour_indexes = get_neighbour(index.name, radius)
    neighbour_df = dataframe.loc[
        list(set(neighbour_indexes).intersection(dataframe.index)), :
    ]
    return [
        neighbour_df.mean(),
        neighbour_df.min(),
        neighbour_df.max(),
        neighbour_df.std(),
    ]


def train_model_with_cross_validation(
    demand_history_df: pd.DataFrame, forecast_horizon: List[int]
) -> TrainingResult:
    full_stacked_df = feature_engineering(demand_history_df)

    (
        stacked_demand_history_test_df,
        stacked_demand_history_train_df,
    ) = custom_train_test_split(forecast_horizon, full_stacked_df)
    predictor_columns = [
        DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
        DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
        YEAR_INDEX_COLUMN_NAME,
        "value_2_years_ago",
        # "neighbour_mean",
        # "neighbour_max",
        # "neighbour_min",
        # "neighbour_std",
    ]
    target_column = ["value"]

    feature_space_steps = [
        (
            "column_selector",
            ColumnTransformer(
                [
                    ("selector", "passthrough", predictor_columns),
                ],
                remainder="drop",
            ),
        ),
        ("min_max_scaler", MinMaxScaler()),
    ]

    feature_space_pipe = Pipeline(feature_space_steps)

    X_train = feature_space_pipe.fit_transform(stacked_demand_history_train_df)
    groups = stacked_demand_history_train_df[DEMAND_POINT_INDEX_COLUMN_NAME].values
    y_train = np.log1p(stacked_demand_history_train_df[target_column]).values.reshape(
        -1
    )

    grid = {
        "xgboost__max_depth": randint(low=3, high=20),
        "xgboost__gamma": uniform(loc=1, scale=9),
        "xgboost__reg_alpha": uniform(loc=40, scale=180),
        "xgboost__reg_lambda": uniform(loc=0, scale=1),
        "xgboost__n_estimators": randint(low=100, high=2000),
        "xgboost__random_state": [42],
    }

    steps = [("xgboost", XGBRegressor())]
    pipe = Pipeline(steps)

    cv = GroupKFold(n_splits=3)
    cv.get_n_splits(X_train, y_train, groups=groups)

    rf_cv = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=grid,
        cv=cv.get_n_splits(X_train, y_train, groups=groups),
        verbose=2,
        n_iter=10,
    )
    rf_cv.fit(X_train, y_train)

    best_pipe = pipe.set_params(**rf_cv.best_params_).fit(X_train, y_train)

    train_pred = pd.Series(
        best_pipe.predict(X_train), index=stacked_demand_history_train_df.index
    )
    print("Score on Train:", mean_absolute_error(train_pred, y_train))

    X_test = feature_space_pipe.transform(stacked_demand_history_test_df)
    y_test = np.log1p(stacked_demand_history_test_df[target_column]).values.reshape(-1)
    test_pred = pd.Series(
        best_pipe.predict(X_test), index=stacked_demand_history_test_df.index
    )
    print("Score on Test:", mean_absolute_error(test_pred, y_test))

    original_y_test = stacked_demand_history_test_df[target_column]
    exp_test_pred = pd.Series(
        np.exp(best_pipe.predict(X_test)), index=stacked_demand_history_test_df.index
    )
    print(
        "Score on Test with exp MAE:",
        mean_absolute_error(exp_test_pred, original_y_test),
    )
    print(
        "Score on Test with exp RMSE:",
        mean_squared_error(exp_test_pred, original_y_test, squared=False),
    )
    pred = stacked_demand_history_test_df.copy()
    pred["pred"] = exp_test_pred
    pred = pred.set_index(
        [
            DEMAND_POINT_INDEX_COLUMN_NAME,
            DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
            DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
            YEAR_INDEX_COLUMN_NAME,
        ]
    )

    forecaster_training_result = ForecasterTrainingResult(
        predictors_transformation_pipeline=feature_space_pipe,
        model_pipeline=best_pipe,
        target_transformation=np.log1p,
    )

    return forecaster_training_result


def custom_train_test_split(
    forecast_horizon: List[int], full_stacked_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stacked_demand_history_train_df = full_stacked_df[
        full_stacked_df[YEAR_INDEX_COLUMN_NAME].isin(
            set(full_stacked_df[YEAR_INDEX_COLUMN_NAME]).difference(forecast_horizon)
        )
    ]
    stacked_demand_history_test_df = full_stacked_df[
        full_stacked_df[YEAR_INDEX_COLUMN_NAME].isin(forecast_horizon)
    ]
    return stacked_demand_history_test_df, stacked_demand_history_train_df


def feature_engineering(
    demand_history_df: pd.DataFrame, mode: str = "training"
) -> pd.DataFrame:
    stacked_demand_history_df = demand_history_df.set_index(
        [
            DEMAND_POINT_INDEX_COLUMN_NAME,
            DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
            DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
        ]
    ).stack(0)
    stacked_demand_history_df.index.names = [
        DEMAND_POINT_INDEX_COLUMN_NAME,
        DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
        DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
        YEAR_INDEX_COLUMN_NAME,
    ]
    stacked_demand_history_df.name = VALUE_COLUMN_NAME
    stacked_demand_history_df = stacked_demand_history_df.to_frame()
    last_year_index = list(
        map(
            lambda index_tuple: (
                index_tuple[0],
                index_tuple[1],
                index_tuple[2],
                str(int(index_tuple[3]) + 2),
            ),
            stacked_demand_history_df.index.values,
        )
    )
    one_year_shifted = stacked_demand_history_df.copy()
    one_year_shifted.index = pd.MultiIndex.from_tuples(
        last_year_index, names=stacked_demand_history_df.index.names
    )
    one_year_shifted.columns = ["value_2_years_ago"]

    neighbour_df = one_year_shifted.copy()
    # computed_features = np.array(
    #     one_year_shifted.progress_apply(lambda x: compute_neighbour(x, one_year_shifted, 2), axis=1).values.tolist()
    # )
    # neighbour_df["neighbour_mean"] = computed_features[:, 0]
    # neighbour_df["neighbour_min"] = computed_features[:, 1]
    # neighbour_df["neighbour_max"] = computed_features[:, 2]
    # neighbour_df["neighbour_std"] = computed_features[:, 3]
    # neighbour_df["neighbour_std"] = neighbour_df["neighbour_std"].fillna(0)

    full_stacked_df = pd.concat([stacked_demand_history_df, neighbour_df], axis=1)
    if mode == "training":
        full_stacked_df = full_stacked_df.dropna(how="any")
    full_stacked_df = full_stacked_df.reset_index()
    full_stacked_df[YEAR_INDEX_COLUMN_NAME] = full_stacked_df[
        YEAR_INDEX_COLUMN_NAME
    ].astype(float)
    return full_stacked_df


def forecast(
    demand_df: pd.DataFrame, forecast_horizon: List[int], forecaster: TrainingResult
) -> pd.DataFrame:
    full_stacked_df = feature_engineering(demand_df, mode="inference")

    full_forecast_stacked_df = full_stacked_df[
        full_stacked_df[YEAR_INDEX_COLUMN_NAME].isin(forecast_horizon)
    ]

    prediction_feature_space = forecaster.predictors_transformation_pipeline.transform(
        full_forecast_stacked_df
    )

    predictions_values = np.exp(
        forecaster.model_pipeline.predict(prediction_feature_space)
    )
    predictions_df = pd.DataFrame(
        {
            DEMAND_POINT_INDEX_COLUMN_NAME: full_forecast_stacked_df[
                DEMAND_POINT_INDEX_COLUMN_NAME
            ],
            YEAR_INDEX_COLUMN_NAME: full_forecast_stacked_df[YEAR_INDEX_COLUMN_NAME],
            VALUE_COLUMN_NAME: predictions_values,
        }
    )

    unstacked_predictions_df = (
        predictions_df.set_index(
            [DEMAND_POINT_INDEX_COLUMN_NAME, YEAR_INDEX_COLUMN_NAME]
        )
        .unstack([YEAR_INDEX_COLUMN_NAME])
        .reset_index()
    )

    unstacked_predictions_df.columns = [
        DEMAND_POINT_INDEX_COLUMN_NAME
    ] + forecast_horizon

    return unstacked_predictions_df


if __name__ == "__main__":
    from tqdm import tqdm

    tqdm.pandas()

    data_path = Path(__file__).parent.parent.parent / "data"
    result_path = data_path / "result"

    # --------------------------------- #
    #             Training              #
    # --------------------------------- #
    demand_history_df = pd.read_csv(data_path / "Demand_History.csv")

    training_forecast_horizon = [2017, 2018]

    forecaster_training_result = train_model_with_cross_validation(
        demand_history_df, training_forecast_horizon
    )

    # --------------------------------- #
    #              Predict              #
    # --------------------------------- #
    forecast_horizon = [2019, 2020]
    demand_to_predict_df = demand_history_df[
        ["demand_point_index", "x_coordinate", "y_coordinate", "2017", "2018"]
    ]

    forecast_df = forecast(
        demand_to_predict_df, forecast_horizon, forecaster_training_result
    )

    # --------------------------------- #
    #               Optim               #
    # --------------------------------- #
    demand_history: pd.DataFrame = pd.read_csv(data_path / "Demand_History.csv")
    demand_forecast = forecast_df.copy()
    existing_infrastructure: pd.DataFrame = pd.read_csv(
        data_path / "exisiting_EV_infrastructure_2018.csv"
    )

    data_model = create_data_model(
        demand_forecast, demand_history, existing_infrastructure, data_path
    )

    all_years_result = apply_scip_optimizer(
        data_model, show_output=True, gap_limit=0.0, limit_in_time=10 * 60 * 1000
    )

    # --------------------------------- #
    #             Submission            #
    # --------------------------------- #
    submission_dataframe = get_scoring_dataframe(all_years_result)

    submission_dataframe.to_csv(
        result_path / f"submission_dataframe_{evse.__version__}.csv", index=False
    )
