import datetime
from pathlib import Path
from typing import Literal, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

import evse
from evse.const import (
    DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
    DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
    DEMAND_POINT_INDEX_COLUMN_NAME,
    RFR,
    SPL,
    SPD,
    VALUE_COLUMN_NAME,
    YEAR_INDEX_COLUMN_NAME,
)
from evse.forecast import ForecasterTrainingResult, TrainingResult
from evse.optimisation import create_data_model
from evse.optimisation._forecaster import forecaster_estimator, SplineEstimator
from evse.optimisation.ortools.scip import apply_scip_optimizer
from evse.scoring import get_scoring_dataframe


def train_model_with_cross_validation(
    predictor_columns: List[str],
    demand_history_df: pd.DataFrame,
    forecast_horizon: List[int],
    forecaster: Literal[RFR, SPL, SPD],
    use_spline= False,
    **kwargs
) -> TrainingResult:
    # TODO: Use constant after rebasing
    full_stacked_df = feature_engineering(demand_history_df)

    (
        stacked_demand_history_test_df,
        stacked_demand_history_train_df,
    ) = custom_train_test_split(forecast_horizon, full_stacked_df)

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
        ("add_splines", kwargs.get("splinet_fitted") if use_spline else FunctionTransformer()),

    ]

    feature_space_pipe = Pipeline(feature_space_steps)

    X_train = feature_space_pipe.fit_transform(stacked_demand_history_train_df)
    groups = stacked_demand_history_train_df[DEMAND_POINT_INDEX_COLUMN_NAME].values
    y_train = stacked_demand_history_train_df[target_column].values.reshape(
        -1
    )

    forecasterarguments = {"predictor_columns": predictor_columns}

    steps = [("forecaster", forecaster_estimator(forecaster, **forecasterarguments))]
    pipe = Pipeline(steps)

    if forecaster["cv"]:

        grid = {
            "forecaster__n_estimators": [10, 20, 100, 200],
            "forecaster__max_features": ["sqrt", "log2"],
            "forecaster__max_depth": [2, 3, 7, 8, 20, 30],
            "forecaster__random_state": [42],
        }

        cv = GroupKFold(n_splits=10)
        cv.get_n_splits(X_train, y_train, groups=groups)

        rf_cv = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            cv=cv.get_n_splits(X_train, y_train, groups=groups),
            verbose=2,
        )

        rf_cv.fit(X_train, y_train)
        print(f"Best Parameters resulting from GridSearch:\n{rf_cv.best_params_}")

        best_pipe = pipe.set_params(**rf_cv.best_params_).fit(X_train, y_train)
    else:
        best_pipe = pipe.fit(X_train, y_train)

    intermediary = best_pipe.predict(X_train)
    train_pred = pd.Series(
        best_pipe.predict(X_train), index=stacked_demand_history_train_df.index
    )
    print("Score on Train:", mean_absolute_error(train_pred, y_train))

    X_test = feature_space_pipe.transform(stacked_demand_history_test_df)
    y_test = stacked_demand_history_test_df[target_column].values.reshape(-1)
    test_pred = pd.Series(
        best_pipe.predict(X_test), index=stacked_demand_history_test_df.index
    )
    print("Score on Test:", mean_absolute_error(test_pred, y_test))

    original_y_test = stacked_demand_history_test_df[target_column]
    exp_test_pred = pd.Series(
        best_pipe.predict(X_test), index=stacked_demand_history_test_df.index
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

    # Fit on whole data
    full_X_train = feature_space_pipe.fit_transform(full_stacked_df)
    full_y_train = full_stacked_df["value"]
    best_pipe = best_pipe.fit(full_X_train, full_y_train)

    forecaster_training_result = ForecasterTrainingResult(
        predictors_transformation_pipeline=feature_space_pipe,
        model_pipeline=best_pipe,
        target_transformation=np.log1p,
    )

    return forecaster_training_result, pred


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
    full_stacked_df = pd.concat([stacked_demand_history_df, one_year_shifted], axis=1)
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

    predictions_values = (
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
    data_path = Path(__file__).parent.parent.parent / "data"
    result_path = data_path / "result"

    demand_history_df = pd.read_csv(data_path / "Demand_History.csv")
    demand_history_df = demand_history_df[demand_history_df.columns.difference(["2010", "2011"])]


    a, b = (
        pd.read_csv("/Users/deirdree.polak/Documents/Repos/EVSE/data/result/submission_dataframe_0.2.0_RandomForestRegressor_2022-.csv"),
        pd.read_csv("/Users/deirdree.polak/Documents/Repos/EVSE/data/result/submission_dataframe_0.2.0_Spline_2022-09-2.csv")

    )


    # --------------------------------- #
    #             INPUTS                #
    # --------------------------------- #

    forecaster = SPL  # Choice in evse.constants: RFR, SPL, SPD

    # --------------------------------- #
    #             Training              #
    # --------------------------------- #
    demand_history_df = pd.read_csv(data_path / "Demand_History.csv")

    training_forecast_horizon = [2018]

    predictor_columns = [
        DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
        DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
        YEAR_INDEX_COLUMN_NAME,
        "value_2_years_ago",
    ]

    forecaster_training_result, predy = train_model_with_cross_validation(
        predictor_columns, demand_history_df, training_forecast_horizon, forecaster
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
    ).clip(lower=0) # clip values less than 0

    # --------------------------------- #
    #           TRAIN RFR               #
    # --------------------------------- #

    # predictor_columns = [
    #     DEMAND_HISTORY_X_COORDINATE_COLUMN_NAME,
    #     DEMAND_HISTORY_Y_COORDINATE_COLUMN_NAME,
    #     YEAR_INDEX_COLUMN_NAME,
    #     "value_2_years_ago",
    #     "spline_forecast"
    # ]
    training_forecast_horizon = [2018]
    forecaster = RFR  # Choice in evse.constants: RFR, SPL, SPD

    forecaster_training_result, predy = train_model_with_cross_validation(
        predictor_columns, demand_history_df, training_forecast_horizon,
        forecaster, use_spline=True, **{"splinet_fitted": forecaster_training_result.model_pipeline}
    )
    demand_to_predict_df = demand_history_df[
        ["demand_point_index", "x_coordinate", "y_coordinate", "2017", "2018"]
    ]

    forecast_df_ = forecast(
        demand_to_predict_df, forecast_horizon, forecaster_training_result
    )


    # --------------------------------- #
    #               Optim               #
    # --------------------------------- #
    demand_history: pd.DataFrame = pd.read_csv(data_path / "Demand_History.csv")
    demand_forecast = forecast_df_.copy()
    existing_infrastructure: pd.DataFrame = pd.read_csv(
        data_path / "exisiting_EV_infrastructure_2018.csv"
    )

    data_model = create_data_model(
        demand_forecast, demand_history, existing_infrastructure, data_path
    )

    all_years_result = apply_scip_optimizer(
        data_model, show_output=True, gap_limit=0.0, limit_in_time=None# 1000 * 60 * 20
    )

    # --------------------------------- #
    #             Submission            #
    # --------------------------------- #
    submission_dataframe = get_scoring_dataframe(all_years_result)

    submission_dataframe.to_csv(
        result_path
        / f"submission_dataframe_{evse.__version__}_{forecaster['name']}_{datetime.datetime.now()}.csv",
        index=False,
    )


